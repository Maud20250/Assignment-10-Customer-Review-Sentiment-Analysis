# 🛒 Amazon Alexa Reviews — Sentiment Analysis

**Assignment 10 | Natural Language Processing**  
**Student:** Jean Billa  
**Institution:** Willis College — Diploma in Data Analysis and Artificial Intelligence  
**Date:** June 2026

---

## 📌 Project Overview

This project applies a full NLP pipeline to classify Amazon Alexa customer reviews as **positive** or **negative**. It covers data preprocessing, feature engineering (Bag of Words, TF-IDF, word embeddings), traditional machine learning classifiers, and sentiment analysis using a pre-trained Large Language Model (DistilBERT).

**Dataset:** [Amazon Alexa Reviews — Kaggle](https://www.kaggle.com/sid321axn/amazon-alexa-reviews)  
- 3,150 customer reviews for Echo, Echo Dot, and other Alexa devices  
- Target variable: `feedback` (1 = Positive, 0 = Negative)  
- ~91.8% positive reviews (imbalanced dataset)

---

## 🎯 Objectives

- Preprocess and clean raw customer review text
- Engineer features using Bag of Words, TF-IDF, and word embeddings
- Visualize semantic relationships with t-SNE and PCA
- Train and evaluate traditional classifiers (Logistic Regression, SVM, Random Forest)
- Fine-tune DistilBERT for sentiment classification
- Compare model performance using accuracy, precision, recall, F1-score, and ROC-AUC

---

## 📁 Repository Structure

```
├── assignment10_sentiment_analysis.ipynb   # Main Google Colab notebook
├── README.md                               # This file
└── report/
    └── assignment10_report.pdf             # 2-page written report
```

---

## 🔧 How to Run the Notebook

### Option 1 — Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Click **File → Open notebook → GitHub**
3. Paste this repository URL and select `assignment10_sentiment_analysis.ipynb`
4. Upload `amazon_alexa.tsv` to your Google Drive
5. Update the file path in **Task 1** if loading from Drive:
   ```python
   df = pd.read_csv('/content/drive/MyDrive/amazon_alexa.tsv', sep='\t')
   ```
6. Run all cells in order: **Runtime → Run all**

> **Note:** The notebook also includes a direct URL loader as fallback — no manual upload needed in most cases.

### Option 2 — Local Environment

```bash
# 1. Clone the repository
git clone https://github.com/Maud20250/<repo-name>.git
cd <repo-name>

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn torch transformers

# 3. Launch Jupyter
jupyter notebook assignment10_sentiment_analysis.ipynb
```

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualization |
| `scikit-learn` | ML models, vectorization, metrics |
| `torch` | PyTorch backend for BERT |
| `transformers` | Hugging Face DistilBERT |

Install all at once:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch transformers
```

---

## 📊 Models Implemented

| Model | Vectorization | Tuning |
|---|---|---|
| Logistic Regression | TF-IDF | GridSearchCV (5-fold CV) |
| Linear SVM | TF-IDF | GridSearchCV (5-fold CV) |
| Random Forest | Bag of Words + One-Hot | 100 estimators |
| DistilBERT (LLM) | Tokenizer (max_len=128) | Fine-tuned 2 epochs |

---

## 📈 Key Results

- **Best traditional model:** Logistic Regression (TF-IDF) — highest ROC-AUC and best generalization
- **Random Forest** showed moderate overfitting (large train–test accuracy gap)
- **DistilBERT** demonstrated the power of transfer learning even on a small fine-tuning subset
- Dataset imbalance (~92% positive) makes **F1-score** and **ROC-AUC** more meaningful than raw accuracy

---

## 📄 Report

The written report (`assignment10_report.pdf`) covers:
1. Dataset introduction and preprocessing steps
2. Feature engineering methodology
3. Model evaluation results (tables and metrics)
4. Deployment strategy
5. Conclusion and recommendations
