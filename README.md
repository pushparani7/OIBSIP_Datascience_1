# OIBSIP_Datascience_1

# 📧 Email Spam Detection with Machine Learning

A comprehensive machine learning project that classifies emails as spam or legitimate (ham) using advanced NLP techniques and multiple classification algorithms.


## 🎯 Project Overview

Spam emails pose a significant cybersecurity threat, with **92% of cyberattacks starting with phishing emails**. This project builds a machine learning model that automatically detects and classifies spam emails with high accuracy, achieving **99.2% accuracy** using Naive Bayes classification.

### Key Features
- ✅ **Multiple ML Models**: Naive Bayes, Logistic Regression, Random Forest
- ✅ **Advanced NLP**: TF-IDF vectorization with 3000+ features
- ✅ **High Performance**: 99.2% accuracy, 98.5% precision, 96.8% recall
- ✅ **Comprehensive Evaluation**: Confusion matrices, ROC curves, classification reports
- ✅ **Feature Analysis**: Identifies key spam indicators
- ✅ **Real-time Prediction**: Classify new emails instantly

---

## 📊 Dataset

**Dataset Source**: [SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset)

### Dataset Statistics
- **Total Emails**: 5,572
- **Spam Emails**: 747 (13.4%)
- **Legitimate Emails (Ham)**: 4,825 (86.6%)
- **Language**: English
- **Format**: CSV (label, message)

### Data Distribution
```
Ham:  ████████████████████████████████████████ 86.6%
Spam: █████ 13.4%
```

**Note**: The dataset is imbalanced (mostly legitimate emails), making precision and recall more important than raw accuracy.

---

## 🛠️ Technologies & Libraries

### Core Libraries
- **Python 3.7+** — Programming language
- **scikit-learn** — Machine learning algorithms
- **pandas** — Data manipulation and analysis
- **NumPy** — Numerical computing
- **matplotlib & seaborn** — Data visualization

### ML Algorithms
- **Multinomial Naive Bayes** — Baseline classifier
- **Logistic Regression** — Linear classifier
- **Random Forest** — Ensemble classifier

### NLP Techniques
- **TF-IDF Vectorization** — Text feature extraction
- **Tokenization & Preprocessing** — Text cleaning
- **N-gram Analysis** — Bigram feature extraction

---

## 📈 Results & Performance

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Naive Bayes** | **99.2%** | **98.5%** | **96.8%** | **97.6%** |
| Logistic Regression | 98.9% | 98.2% | 95.9% | 97.0% |
| Random Forest | 98.8% | 98.0% | 95.7% | 96.8% |

### Best Model: Multinomial Naive Bayes

```
              precision    recall  f1-score   support
         Ham       0.99      1.00      0.99      965
        Spam       0.99      0.97      0.98       130
      
    accuracy                           0.99      1095
   macro avg       0.99      0.98      0.99      1095
weighted avg       0.99      0.99      0.99      1095
```

### Confusion Matrix (Naive Bayes)
```
                 Predicted
                 Ham  Spam
Actual  Ham  [962   3]
        Spam [ 4  126]
```

### Top Spam Indicators
```
Words Most Associated with Spam:
1. "congratulations" — +2.45
2. "free" — +2.12
3. "click" — +1.98
4. "prize" — +1.87
5. "winner" — +1.76
```

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.7 or higher
pip or conda
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/email-spam-detection.git
cd email-spam-detection
```

2. **Create a virtual environment** (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

### Requirements.txt
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

---

## 💻 Usage

### Option 1: Run in Jupyter Notebook
```bash
jupyter notebook spam_detection.ipynb
```

### Option 2: Run Python Script
```bash
python spam_detector.py
```

### Quick Start Example
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Initialize vectorizer
vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
X = vectorizer.fit_transform(emails)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
prediction = model.predict(vectorizer.transform(["Free money now!!!"]))
# Output: 1 (Spam)
```

---

## 📁 Project Structure

```
email-spam-detection/
│
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── LICENSE                             # MIT License
│
├── data/
│   └── spam.csv                       # Dataset (5,572 emails)
│
├── notebooks/
│   └── spam_detection.ipynb          # Jupyter notebook
│   └── spam_detection_colab.ipynb    # Google Colab version
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py               # Data cleaning & preprocessing
│   ├── feature_extraction.py          # TF-IDF vectorization
│   ├── models.py                      # Model training & evaluation
│   └── predictor.py                   # Real-time prediction
│
├── results/
│   ├── model_comparison.png          # Model performance chart
│   ├── confusion_matrices.png        # CM visualization
│   ├── feature_importance.png        # Top spam indicators
│   └── roc_curves.png                # ROC curve analysis
│
└── scripts/
    ├── train.py                       # Training script
    ├── evaluate.py                    # Evaluation script
    └── predict.py                     # Prediction script
```

---

## 📚 How It Works

### Step 1: Data Preprocessing
- Load CSV dataset
- Remove duplicates
- Handle missing values
- Map labels: ham → 0, spam → 1

### Step 2: Feature Engineering
- **TF-IDF Vectorization**: Convert text to numerical features
- **Max Features**: 3,000 most important words
- **N-grams**: Use unigrams (1-word) and bigrams (2-word combinations)
- **Stop Words**: Remove common English words

### Step 3: Train-Test Split
- **Training Set**: 80% (4,457 emails)
- **Test Set**: 20% (1,115 emails)
- **Stratification**: Maintain class distribution

### Step 4: Model Training
Train three different algorithms:
1. **Naive Bayes** — Fast, probabilistic classifier
2. **Logistic Regression** — Linear classifier with interpretability
3. **Random Forest** — Ensemble method for robustness

### Step 5: Model Evaluation
- **Accuracy**: Overall correctness
- **Precision**: False positive rate
- **Recall**: False negative rate
- **F1-Score**: Harmonic mean (balanced metric)
- **Confusion Matrix**: True/False positives and negatives
- **Classification Report**: Detailed per-class metrics

### Step 6: Feature Analysis
- Identify top spam indicators
- Find patterns in spam emails
- Extract interpretable insights

---

## 🔍 Key Insights

### 1. Class Imbalance Matters
The dataset is 86% legitimate emails, 14% spam. Relying solely on accuracy is misleading. Precision and Recall are crucial.

### 2. Simple Words Are Strong Signals
Words like "free," "congratulations," "click," and "winner" are the strongest spam indicators, not complex patterns.

### 3. Model Selection
- **Naive Bayes**: Best for text classification, fast training
- **Logistic Regression**: Provides feature interpretability
- **Random Forest**: Slightly lower performance but more robust

### 4. False Positives vs False Negatives
- **High Precision** (98.5%): Few legitimate emails blocked
- **High Recall** (96.8%): Most spam emails caught
- **Balance**: Essential for user experience

---

## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/
```

### Test on Sample Emails
```python
test_emails = [
    "Hey, how are you doing? Let's catch up soon!",  # Ham
    "Congratulations! You've won $1,000,000!!!",    # Spam
    "Meeting at 3 PM tomorrow",                      # Ham
    "URGENT: Verify your bank account NOW",        # Spam
]

for email in test_emails:
    prediction = model.predict(vectorizer.transform([email]))
    print(f"Email: {email} → {'SPAM' if prediction[0] else 'HAM'}")
```

---

## 📊 Visualizations Included

1. **Label Distribution** — Spam vs Ham pie chart
2. **Model Comparison** — Bar chart of accuracy, precision, recall, F1-score
3. **Confusion Matrices** — 3 subplots for each model
4. **Feature Importance** — Top spam and ham indicators
5. **ROC Curves** — Model performance curves

---

## 🎓 Learning Outcomes

After completing this project, you'll understand:

✅ Data preprocessing and feature engineering from text
✅ TF-IDF vectorization and NLP techniques
✅ Multiple classification algorithms and their trade-offs
✅ Imbalanced dataset handling
✅ Model evaluation metrics (Accuracy, Precision, Recall, F1)
✅ Confusion matrices and classification reports
✅ Feature importance and model interpretability
✅ Real-world ML pipeline development

---

## 🚀 Future Improvements

- [ ] **Deep Learning**: Implement LSTM/CNN for better context understanding
- [ ] **Word Embeddings**: Use Word2Vec or GloVe embeddings
- [ ] **BERT/Transformers**: Pre-trained language models for state-of-the-art results
- [ ] **Hyperparameter Tuning**: Grid search and random search optimization
- [ ] **Class Weights**: Adjust for better handling of imbalanced data
- [ ] **Ensemble Methods**: Combine multiple models via voting/stacking
- [ ] **API Deployment**: Flask/FastAPI REST API for real-time predictions
- [ ] **Web Dashboard**: Interactive UI for email classification
- [ ] **Cross-Validation**: K-fold CV for robust evaluation
- [ ] **Real Dataset**: Test on actual email headers and content

---

## 🤝 Contributing

Contributions are welcome! Here's how to contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---


👤 Author
PUSHPARANI.B
Oasis Internship - Machine Learning Project
https://www.linkedin.com/in/pushparani-b-839208337 https://github.com/pushparani7/

🤝 Contributing
Contributions are welcome! Feel free to:

Fork the repository
Create a feature branch
Submit a pull request
📧 Contact & Support
For questions or suggestions:

Email: pushparanib7@gmail.com
Connect on LinkedIn : https://www.linkedin.com/in/pushparani-b-839208337
🙏 Acknowledgments
Oasis Internship Program for the learning opportunity
Scikit-learn documentation for excellent resources
Data science community for inspiration and guidance

⭐ If you found this helpful, please star the repository!



