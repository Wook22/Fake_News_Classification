
# 📰 Logistic Regression Analysis on BuzzFeed Data

This project performs a comprehensive binary classification analysis using logistic regression on BuzzFeed article data. The goal is to distinguish between real and fake articles based on textual features.

## 📁 Dataset

The data comes from a cleaned and processed version of BuzzFeed news articles. The dataset includes:

* A binary response variable `real_fake`
* Word count features: `word_1`, `word_2`, ..., `word_30`
* Metadata including `authors`, `publish_date`, and derived features

The final working dataset used is `df_drop`, a cleaned version where:

* Rows with missing `publish_date` or empty `authors` are removed
* The number of word features was reduced to the top 30 to stabilize the model

## 🔍 Modeling Approach

A logistic regression model was fit to predict whether an article is real or fake:

```r
model_best <- glm(real_fake ~ ., data = df_drop, family = binomial)
```

### ✅ Confidence Intervals

Confidence intervals for the multiplicative effects of predictors were computed:

```r
confint(model_best)
```

### 📊 Prediction and Classification

Predicted probabilities were generated and various cut-off points were explored to evaluate classification performance:

```r
cutoffs <- seq(0.1, 0.9, by = 0.1)
# Loop through and print confusion matrices
```

### 🧪 ROC Curve and AUC

Model performance was assessed using ROC curve and AUC:

```r
library(pROC)
roc_obj <- roc(df_drop$real_fake, predict(model_best, type = "response"))
plot(roc_obj)
auc(roc_obj)
```

The best threshold for classification was determined using Youden’s Index.

### 🔄 Cross-Validation

Model reliability was validated using:

* **LOOCV (Leave-One-Out Cross-Validation)** via `cv.glm()`
* **k-Fold Cross-Validation (k = 10)** via the `caret` package

```r
cv.glm(df_drop, model_best, K = nrow(df_drop))$delta  # LOOCV
```

### 🔁 Link Function Comparison

Alternative models were tested using different link functions:

* **Logit (default)**
* **Probit**
* **Identity**

```r
glm(..., family = binomial(link = "probit"))
glm(..., family = binomial(link = "identity"))
```

Their AUC scores were compared to identify the best fit.

### 📊 Contingency Table Analysis (if applicable)

Chi-squared and likelihood ratio (G²) tests were conducted on grouped categorical data (if present):

```r
chisq.test(table(df_drop$real_fake, df_drop$some_categorical_var))
```

## 📈 Summary

* The reduced logistic model (`model_best`) performed well with 30 predictors
* AUC and cross-validation metrics indicated robust classification
* The logit link performed slightly better than probit and identity for this dataset
* ROC curve visualization and optimal cut-off selection improved model interpretability

## 📦 Requirements

Install the following R packages:

```r
install.packages(c("dplyr", "pROC", "boot", "caret", "brglm2"))
```

## 📁 File Structure

```
├── README.md
├── buzzfeed_cleaning.R        # Data cleaning and preprocessing
├── model_fit.R                # Logistic regression modeling
├── model_eval.R               # ROC, AUC, CV, and confusion matrix
└── buzzfeed_data.csv          # Cleaned input dataset
```

## ✍️ Author

JongWook Choe
Master's Student in Statistics

---
