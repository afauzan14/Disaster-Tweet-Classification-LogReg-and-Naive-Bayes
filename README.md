# Disaster Tweet Classification and Emergency Event Extraction

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![spaCy](https://img.shields.io/badge/spaCy-09A3D5?logo=spacy&logoColor=white)](https://spacy.io/)
[![Jupyter Notebook](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter&logoColor=white)](UTS_NLP_Kelompok1.ipynb)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20Disaster%20Tweets-blue)](https://www.kaggle.com/c/nlp-getting-started)
[![ROC-AUC](https://img.shields.io/badge/Metric-0.91%20ROC--AUC-success)]()

---

## Project Overview

During natural disasters and humanitarian crises, real-time social media streams provide immediate situational awareness. However, disaster-related keywords (such as *fire, collapse, deluge, ablaze*) are frequently used metaphorically in colloquial language, producing high volumes of noise for emergency responders.

This project develops an end-to-end Natural Language Processing (NLP) pipeline that:
1. **Differentiates Genuine Crisis Reports from Metaphors:** Classifies text data into genuine disaster events versus non-disaster chatter using TF-IDF vectorization and hyperparameter-tuned classical machine learning models.
2. **Evaluates Classification Performance:** Conducts a benchmark between **Logistic Regression** and **Multinomial Naive Bayes**, evaluated via Stratified K-Fold Cross-Validation and ROC-AUC analysis.
3. **Extracts Critical Incident Metadata:** Integrates **Named Entity Recognition (NER)** via `spaCy` to automatically parse actionable incident intelligence (**Event type, Date, Time, and Location**) from verified disaster tweets.

---

## Dataset and Preprocessing Pipeline

* **Dataset:** 7,613 annotated Twitter records sourced from the Kaggle *Natural Language Processing with Disaster Tweets* challenge.
* **Exploratory Insights:** Real disaster announcements exhibit distinct linguistic conciseness—containing shorter, direct syntax compared to lengthy conversational non-disaster tweets.
* **Text Normalization:**
  * Cleaned URLs, account handles (`@mention`), special punctuation characters, numeric strings, and emojis.
  * Standardized lowercase transformations and filtered standard English stopwords.
* **Feature Representation:**
  * Extracted unigram and bigram token features using **TF-IDF Vectorization** (`sublinear_tf=True`), penalizing common lexical tokens while boosting domain-discriminative vocabulary.
* **Data Partitioning:** 80% Training set (6,090 samples) and 20% Holdout Test set (1,523 samples).

---

## Model Optimization and Hyperparameter Tuning

Both classifiers were tuned using **GridSearchCV with Stratified 5-Fold Cross-Validation**, optimizing specifically for the minority disaster class F1-Score:

* **Tuned Logistic Regression:**
  * Optimal Configuration: `C = 10`, `penalty = 'l2'`, `class_weight = 'balanced'`, `solver = 'liblinear'`.
  * Best Cross-Validation F1-Score: **0.7803**.
* **Tuned Multinomial Naive Bayes:**
  * Optimal Configuration: `alpha = 0.01` (Laplace smoothing), `class_prior = [0.5, 0.5]`, `fit_prior = True`.
  * Best Cross-Validation F1-Score: **0.8001**.

---

## Comparative Performance Evaluation

Performance metrics evaluated on the 1,523 unseen holdout test samples:

| Model Architecture | Accuracy | Disaster Precision | Disaster Recall | Disaster F1-Score | ROC-AUC Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | **85.0%** | **0.85** | **0.80** | **0.82** | **0.91** |
| **Logistic Regression** | **83.0%** | **0.80** | **0.80** | **0.80** | **0.89** |

### Key Diagnostic Takeaways
1. **Superior Discriminative Capability:** Multinomial Naive Bayes achieved the top overall accuracy (85.0%) and an exceptional **0.91 ROC-AUC score**, demonstrating reliable decision boundaries with minimal false-positive alarms.
2. **Metaphor Disambiguation:** Error analysis revealed that ambiguous words (e.g., *“ablaze”* used in non-crisis contexts) were more effectively filtered by Naive Bayes due to prior probability weighting across multi-token contexts.

---

## Automated Incident Metadata Extraction (spaCy NER)

To bridge text classification into actionable emergency response, positive disaster predictions are forwarded to an information extraction module built on `spaCy` (`en_core_web_sm`):

| Extracted Entity | Extraction Method | Operational Value for First Responders |
| :--- | :--- | :--- |
| **Incident Event** | `doc.ents (EVENT)` | Identifies the physical hazard (e.g., *wildfire, typhoon, earthquake*). |
| **Temporal Data** | `doc.ents (DATE, TIME)` | Determines exact timelines and urgency of the unfolding incident. |
| **Geographic Location** | `doc.ents (GPE, LOC)` | Maps affected regions to assist rapid dispatch coordination. |

---

## Summary of Technical Contributions

* Developed a lightweight, low-latency text classification pipeline achieving 0.91 ROC-AUC without requiring heavy neural network compute.
* Successfully separated metaphorical social media language from actual hazard emergencies.
* Extended raw classification into a functional intelligence extraction system parsing critical event, temporal, and spatial entities for situational disaster response.
