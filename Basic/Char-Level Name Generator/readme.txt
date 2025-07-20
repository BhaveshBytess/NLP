
# 🧠 Character-Level Name Generator with Gender Conditioning (LSTM)

A powerful, lightweight name generator built using a character-level LSTM model trained on the NLTK Names Corpus. It generates realistic male or female names from scratch, character-by-character, with support for temperature-based sampling and persistent model saving.

---

## 🚀 Features

* ✅ Character-level LSTM sequence modeling
* ✅ Gender conditioning (generate male or female names on demand)
* ✅ Temperature-controlled sampling (diversity vs coherence)
* ✅ Persistent model saving and reloading
* ✅ Reusable modular pipeline
* ✅ Clean and beginner-friendly architecture
* ✅ Built with TensorFlow / Keras and Python

---

## 📂 Dataset

We use the NLTK Names Corpus containing separate lists of male and female names:

* 🔹 `data/names/male.txt`
* 🔹 `data/names/female.txt`

You can load it using NLTK or manually download and save them as text files.

---

## 🏗️ Project Structure

```
📦name-generator
 ┣ 📁data/
 ┃ ┗ 📁names/
 ┃   ┣ male.txt
 ┃   ┗ female.txt
 ┣ 📁models/
 ┃ ┣ namegen_lstm.h5
 ┃ ┗ char_mappings.pkl
 ┣ 📄namegen_pipeline.py
 ┣ 📄train.py
 ┣ 📄generate.py
 ┗ 📄README.md
```

---

## 🧪 Example Outputs

```text
📌 Male: ['Tavion', 'Jorren', 'Antez', 'Kendric', 'Dathon']
📌 Female: ['Linaya', 'Averra', 'Sarela', 'Myra', 'Jenice']
```

---

## 🔧 Usage

### 🏋️ 1. Train the model

```bash
python train.py
```

This will:

* Clean and prepare the data
* Train an LSTM model to learn character sequences
* Save the model and character mappings to disk

---

### 🔮 2. Generate names

After training, run:

```bash
python generate.py
```

This will:

* Load the saved model and mappings
* Generate gender-specific names using temperature sampling

You can adjust:

* Number of names (`n=20`)
* Gender (`gender='F'` or `'M'`)
* Temperature (`temperature=0.8` for more diversity)

---

## ⚙️ Core Concepts

This project helps you learn:

* ✅ Sequence modeling with LSTM
* ✅ Character-level embeddings
* ✅ Conditional text generation
* ✅ Temperature sampling for controlled creativity
* ✅ Reusable ML pipelines

---

## 📌 Requirements

```bash
tensorflow
numpy
pandas
nltk
```

Install them with:

```bash
pip install -r requirements.txt
```

---

## ✅ TODO (Optional Enhancements)

* [ ] Add GUI/Streamlit interface
* [ ] Train on larger, multi-language name datasets
* [ ] Explore Transformer-based name generation (as a next step!)

---

## 👤 Author

Built with ❤️ by \[Your Name]

Feel free to fork, star ⭐, and contribute!

---

Would you like me to save this README to a markdown file or copy it into a textdoc on the side for editing?
