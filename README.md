
---

# **Fake News Classification Using Word Frequency Features**

This project performs **Exploratory Data Analysis (EDA)**, **feature engineering**, and **machine learning modeling** to classify **real vs. fake news** using binary keyword indicators extracted from BuzzFeed news datasets.

The pipeline includes:

* Dataset loading and cleaning
* Media-link feature extraction
* Text preprocessing
* Title/body frequent-word analysis
* Feature engineering (top 20 binary keyword indicators)
* Model training (LR, RF, GB, Bagging, KNN)
* KNN hyperparameter optimization
* Confusion matrix comparison
* Deep learning extension (LSTM)
* Export of final SVM model

---

## **📂 Dataset Sources**

This project uses files from **FakeNewsNet**, specifically the BuzzFeed subset:

* `BuzzFeed_fake_news_content.csv`
* `BuzzFeed_real_news_content.csv`

(Additionally GossipCop and PolitiFact are loaded, but not used in modeling.)

---

## **📘 1. Data Loading**

```python
Buzzfeed_f = pd.read_csv("data/BuzzFeed_fake_news_content.csv")
Buzzfeed_r = pd.read_csv("data/BuzzFeed_real_news_content.csv")
```

The real and fake datasets are **merged** and a `news_type` label is extracted from the `id` field.

---

## **🧹 2. Data Preprocessing**

### **Steps:**

1. Combine real and fake BuzzFeed data
2. Extract target column:

   * `"Real"` → 1
   * `"Fake"` → 0
3. Create binary media indicators:

```python
Buzzfeed_merge['contain_movies'] = ...
Buzzfeed_merge['contain_images'] = ...
```

4. Remove irrelevant columns:

```
id, url, top_img, authors, publish_date, meta_data, canonical_link, movies, images
```

5. Save cleaned dataset:

```python
Buzzfeed_clean.to_csv("data/Buzzfeed_data.csv")
```

---

## **📊 3. Exploratory Data Analysis (EDA)**

### **Source Distribution**

Barplots show news counts per source for real and fake news.

### **Common Sources**

Identify sources appearing in **both** real and fake news.

### **Media-Link Presence**

Countplots for:

* Articles containing **images**
* Articles containing **videos**

### **Frequent Word Extraction**

Using a custom preprocessing pipeline (lowercase → remove punctuation → remove stopwords → stemming → whitespace cleanup), the top words are computed using `CountVectorizer`.

* **Top 20 frequent title words** (fake vs real)
* **Top 30 frequent body words** (fake vs real)

CSV files saved:

```
top1_fake_title.csv
top2_real_title.csv
top3_fake_body.csv
top4_real_body.csv
```

---

## **🧪 4. Feature Engineering**

Using the most frequent words:

* **Top 5 fake-title words**
* **Top 5 real-title words**
* **Top 5 fake-body words**
* **Top 5 real-body words**

Total engineered features: **20 binary indicator columns**.

Example:

```python
Buzzfeed_title["fake_title_hillary"] = ...
Buzzfeed_body["real_body_trump"] = ...
```

Media indicators also included:

* `contain_images`
* `contain_movies`

---

## **🤖 5. ML Modeling (Classical Models)**

Models evaluated:

* **Logistic Regression**
* **Random Forest**
* **Gradient Boosting**
* **Bagging (Decision Tree)**
* **KNN (k=5)**

All models except KNN reached:

```
Accuracy: 0.7027
```

KNN scored:

```
0.5135
```

---

## **🔍 6. KNN Hyperparameter Tuning**

Grid search tested:

* `k = 1–20`
* `weights = uniform, distance`

Best parameters:

```
k = 17
weights = uniform
CV Accuracy ≈ 0.5103
```

A performance plot (`knn_plot.png`) visualizes accuracy vs. k.

---

## **📉 7. Confusion Matrix Comparison**

Four top models (LR, RF, GB, Bagging) produced **identical confusion matrices**:

```
Pred Real | Fake Actual → 11 FP
Pred Fake | Fake Actual → 8 TN
Pred Real | Real Actual → 18 TP
Pred Fake | Real Actual → 0 FN
```

Key takeaway:

* **Recall (Real): 100%**
* **Recall (Fake): 42%**

All misclassifications were **false positives**.

Saved as:

```
confusion_matrices_2x2_comparison.png
```

---

## **🧠 8. Deep Learning (LSTM Model)**

A simple LSTM was trained on news **titles**:

* Tokenization with vocab size = 10,000
* Padding to length 20
* LSTM with 64 units
* Dropout = 0.5

Performance shown after 10 epochs.

---

## **📦 9. Exporting an SVM Text-Classification Model**

The text is preprocessed, vectorized, TF-IDF transformed, and fed into a **linear SVM**.

Saved model:

```
data/svm_body_model.pkl
```

---

## **🏁 Final Notes**

This project demonstrates:

* How minimal binary keyword indicators can separate real vs. fake news
* That engineered features favor real-news detection
* KNN is not suited for sparse indicator features
* Ensemble models generalize best on small datasets
* Deep learning requires much larger datasets for improvement

---

## **📁 Project Structure**

```
├── data/
│   ├── BuzzFeed_real_news_content.csv
│   ├── BuzzFeed_fake_news_content.csv
│   ├── Buzzfeed_data.csv
│   ├── top1_fake_title.csv
│   ├── top2_real_title.csv
│   ├── top3_fake_body.csv
│   ├── top4_real_body.csv
│   ├── svm_body_model.pkl
│
├── confusion_matrices_2x2_comparison.png
├── knn_plot.png
├── EDA6.png
├── EDA7.png
│
└── README.md (this file)
```

---

## **📬 Contact**

**Jong Wook Choe**
Department of Mathematics, Statistical Data Science
San Francisco State University
📧 *[jchoe3@sfsu.edu](mailto:jchoe3@sfsu.edu)*

---


