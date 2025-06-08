import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import contractions
import emoji
from textblob import TextBlob
import unicodedata

# Make sure the required NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize required components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text,
               lowercase=True,
               expand_contractions=True,
               remove_html=True,
               remove_urls=True,
               remove_punct=True,
               remove_numbers=True,
               remove_stopwords=True,
               lemmatize=True,
               remove_emojis=True,
               map_emojis_to_text=False,
               correct_spelling=False):
    """
    Cleans input text string using the following steps:
    1. Unicode normalization & whitespace cleanup
    2. Lowercasing
    3. Expanding contractions
    4. Emoji handling (remove or map to text)
    5. Removing HTML tags
    6. Removing URLs
    7. Removing punctuation
    8. Removing numbers
    9. Tokenization
    10. Stopword removal
    11. Lemmatization
    12. Spelling correction (optional, slow)
    Returns: List of clean tokens
    """
    if not isinstance(text, str):
        return []

    # 1. Unicode normalization & whitespace cleanup
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'\s+', ' ', text).strip()

    # 2. Lowercase
    if lowercase:
        text = text.lower()

    # 3. Expand contractions
    if expand_contractions:
        text = contractions.fix(text)

    # 4. Emoji handling
    if map_emojis_to_text:
        text = emoji.demojize(text)
    elif remove_emojis:
        text = emoji.replace_emoji(text, replace='')

    # 5. Remove HTML tags
    if remove_html:
        text = re.sub(r'<.*?>', '', text)

    # 6. Remove URLs
    if remove_urls:
        text = re.sub(r'http\S+|www\S+', '', text)

    # 7. Remove punctuation
    if remove_punct:
        text = re.sub(r'[^\w\s]', '', text)

    # 8. Remove numbers
    if remove_numbers:
        text = re.sub(r'\d+', '', text)

    # 9. Tokenize
    tokens = word_tokenize(text)

    # 10. Remove stopwords
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stop_words]

    # 11. Lemmatization
    if lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    # 12. Spelling correction (slow)
    if correct_spelling and tokens:
        blob = TextBlob(" ".join(tokens))
        corrected = blob.correct()
        tokens = word_tokenize(str(corrected))

    return tokens

from tqdm import tqdm
tqdm.pandas()  # Enable progress bar on DataFrame operations

def clean_dataframe_column(df, column, **cleaner_kwargs):
    """
    Apply clean_text() to a DataFrame column.
    
    Parameters:
    - df: pandas DataFrame
    - column: column name containing raw text
    - cleaner_kwargs: keyword arguments passed to clean_text()
    
    Returns:
    - df: DataFrame with a new column: column_cleaned
    """
    cleaned_col = f"{column}_cleaned"
    df[cleaned_col] = df[column].progress_apply(lambda x: clean_text(str(x), **cleaner_kwargs))
    return df