# 🌪️ Disaster Tweet Classification using Classical Machine Learning

This project focuses on classifying tweets into **disaster-related** and **non-disaster** categories using classical machine learning approaches. The objective is to analyze textual patterns in social media data and evaluate model performance for early disaster information filtering.

---

## 📌 Project Overview

Social media platforms often become a primary source of real-time information during disaster events. This project aims to build a text classification pipeline to automatically identify disaster-related tweets based on their content.

The analysis includes:
- Exploratory Data Analysis (EDA) on tweet characteristics
- Feature extraction using text vectorization
- Model training and evaluation using classical machine learning methods

---

## 📊 Exploratory Data Analysis

Key observations from the data exploration:
- **Disaster-related tweets tend to be shorter and more direct**
- Non-disaster tweets are generally longer and more conversational
- Tweet length distribution provides useful insight for feature engineering and model behavior

The distribution of tweet character lengths is used to support these findings.

---

## 🤖 Modeling Approach

Two classical machine learning models were evaluated:
- **Logistic Regression**
- **Naive Bayes**

Text data was transformed into numerical features using **TF-IDF vectorization**, allowing the models to capture important word patterns.

---

## 📈 Model Evaluation

Model performance was assessed using the **ROC curve** and **ROC-AUC score**.

**Key results:**
- Naive Bayes achieved an **ROC-AUC score of 0.91**
- The ROC curve indicates a high true positive rate with low false positives
- The model demonstrates strong capability for separating disaster and non-disaster tweets

These results suggest that classical models can be effective for early-stage disaster tweet filtering.

---

## 🧰 Tech Stack

- **Programming Language:** Python  
- **Libraries & Tools:**
  - Scikit-learn
  - Pandas
  - NumPy
  - Matplotlib
- **Techniques:**
  - TF-IDF Vectorization
  - Logistic Regression
  - Naive Bayes
  - ROC & AUC Evaluation

---

## 🎯 Key Takeaways

- Simple classical machine learning models can perform well on text classification tasks
- Tweet length and linguistic structure play an important role in disaster detection
- Naive Bayes provides a strong baseline with efficient performance and interpretability

---

## 📎 Notes

This project is intended as a demonstration of NLP fundamentals, exploratory analysis, and model evaluation using classical machine learning approaches on real-world social media data.
