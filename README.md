# 🧠 BioNER with PubMedBERT

This project demonstrates a Named Entity Recognition (NER) system trained on biomedical text using **PubMedBERT**. It uses deep learning to identify and label biomedical entities in text using IOB tagging format.

---

## 📌 Features

- ✅ Pretrained PubMedBERT from HuggingFace Transformers
- ✅ Fine-tuning on biomedical NER dataset (e.g., AnatEM)
- ✅ Custom PyTorch training loop
- ✅ Inference script to test on new input sentences

---

## 🧪 Technologies

- Python 3.9+
- PyTorch
- HuggingFace Transformers
- Scikit-learn

---

## 🗃️ Dataset

This project uses biomedical NER datasets provided by the [MTL-Bioinformatics-2016 Shared Task](https://github.com/cambridgeltl/MTL-Bioinformatics-2016).

🔗 Direct link to datasets:  
https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data

**Note**: Datasets are not included in this repository due to licensing.

---

## 📁 Project Structure

BioNER-PubMedBERT/
│
├── ner_pubmedbert.py # Training script (steps 1–7)
├── inference.py # Inference script (step 8)
├── model/ # Model artifacts (config, tokenizer, etc.)
├── requirements.txt # Required packages
└── README.md # This file

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt


## 2. Train the model
python ner_pubmedbert.py


## 3. Run inference
python inference.py

You can test with your own sentences by editing the input string in inference.py.


✨ ## Example Output
The → O
thoracic → B-Anatomical_entity
cavity → I-Anatomical_entity
contains → O
heart → B-Anatomical_entity
and → O
lungs → B-Anatomical_entity



👨‍💻 Author
Ahmad Paknezhad
M.Sc. Student in Artificial Intelligence in Medicine
📧 Email: a.s.paknezhad.1@gmail.com
🔗 LinkedIn: https://www.linkedin.com/in/ahmad-sourtiji-paknezhad-353842373/

