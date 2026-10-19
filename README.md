# 📩 Spam Message Classifier Using NLP

A machine learning project that classifies text messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) and the Multinomial Naive Bayes algorithm.

## 🎯 Project Objective

The goal of this project is to build a text classification model that identifies potentially unwanted messages using word-based features and supervised machine learning.

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- CountVectorizer (Bag of Words)
- Multinomial Naive Bayes
- Matplotlib
- Streamlit (for the web application)

## ⚙️ Project Workflow

1. Load and explore the dataset.
2. Clean and validate the messages.
3. Encode labels: Ham = 0 and Spam = 1.
4. Split the data into training and testing sets.
5. Convert text into numerical features using CountVectorizer.
6. Train a Multinomial Naive Bayes classifier.
7. Evaluate performance using accuracy, precision, recall, F1-score, and a confusion matrix.
8. Predict whether new messages are spam or ham.

## 🧠 NLP Techniques

### Bag of Words

CountVectorizer converts text messages into numerical word-count features that a machine learning model can process.

### Multinomial Naive Bayes

A probabilistic classification algorithm that learns patterns in word counts associated with spam and ham messages.

## 📊 Model Evaluation

The model is evaluated on a held-out test dataset using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

*Actual evaluation metrics will be added after running the final version of the notebook.*

## 📁 Project Structure

```text
spam-message-classifier/
├── app.py
├── spam_classifier.ipynb
├── requirements.txt
├── spam_classifier.joblib
└── README.md
```

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/eng-gotam/spam-message-classifier.git
cd spam-message-classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

Ensure that `spam_classifier.joblib` and `app.py` are in the project root directory.

## ⚠️ Limitations

- The model uses word counts rather than deep contextual language understanding.
- Some legitimate messages may be incorrectly classified as spam.
- Predictions depend on the quality and representativeness of the training dataset.
- Estimated probabilities are not guarantees of correctness.

## 🔮 Future Improvements

- Compare Bag of Words with TF-IDF.
- Evaluate Logistic Regression and other classifiers.
- Improve model evaluation and error analysis.
- Deploy the application using Streamlit Community Cloud.

## 👨‍💻 Author

**Gotam Kumar**

BS Artificial Intelligence Student | Sindh Madressatul Islam University (SMIU), Karachi

GitHub: [@eng-gotam](https://github.com/eng-gotam)
Live app: [https://spam-message-classifier-wq7lwreshex4huvrnhzcqe.streamlit.app/]

---

*Built as a practical project while learning Traditional NLP and Machine Learning.*
