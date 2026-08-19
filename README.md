# Telco Customer Churn Prediction

## 🤖 Project Overview

This project develops a machine learning model to predict customer churn for a telecommunications company.

The objective is to identify customers who are at higher risk of leaving so that the business can take proactive retention actions.

The project follows an end-to-end machine learning workflow covering **exploratory data analysis, feature engineering, preprocessing, model comparison, hyperparameter tuning, ROC-AUC evaluation, threshold optimization, model interpretation, and business recommendations.**

---

## 🎯 Business Problem

Customer churn can have a significant impact on a telecom company's revenue and customer base.

The goal of this project is to answer:

* Which customers are more likely to churn?
* What customer characteristics are associated with higher churn risk?
* Which machine learning model performs better?
* How can the classification threshold be adjusted to identify more potential churners?
* Which factors should the business focus on when designing retention strategies?

---

## 📊 Dataset

The dataset contains **7,043 customer records and 21 columns**, covering customer demographics, services, contract information, payment methods, charges, and churn status.

Key variables include:

* Customer ID
* Gender
* Senior Citizen
* Partner
* Dependents
* Tenure
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming Services
* Contract
* Paperless Billing
* Payment Method
* Monthly Charges
* Total Charges
* Churn

The target variable is **Churn**.

---

## 🛠️ Tools & Technologies

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Logistic Regression**
* **Random Forest**
* **GridSearchCV**
* **StandardScaler**
* **OneHotEncoder**
* **Pipeline**
* **ColumnTransformer**
* **Google Colab**

---

## 🔄 Project Workflow

### 1. Data Loading & Initial Inspection

The dataset was loaded using Pandas and inspected to understand:

* Dataset dimensions
* Data types
* Missing values
* Numerical and categorical variables
* Target distribution

The customer ID was removed before machine learning because it does not provide meaningful predictive information.

---

### 2. Exploratory Data Analysis

The analysis examined churn across important customer characteristics.

#### Churn Distribution

The dataset contains:

* **5,174 customers who did not churn**
* **1,869 customers who churned**

This corresponds to an overall churn rate of approximately **26.54%**.

#### Churn by Contract

Month-to-month customers showed the highest churn rate:

* Month-to-month: **42.71%**
* One year: **11.27%**
* Two year: **2.83%**

#### Churn by Internet Service

* Fiber optic: **41.89%**
* DSL: **18.96%**
* No internet service: **7.40%**

#### Churn by Payment Method

Electronic check customers showed the highest churn rate at approximately **45.29%**.

---

## ⚙️ Feature Engineering

A new feature called **AvgMonthlySpend** was created using:

`TotalCharges / tenure`

This feature was added to provide an additional measure of the customer's average spending over their tenure.

Zero-tenure customers were handled so that division by zero would not occur.

---

## 🧹 Data Preprocessing

A Scikit-learn preprocessing pipeline was created to handle numerical and categorical variables.

### Numerical Features

Numerical variables were:

* Median imputed
* Standardized using `StandardScaler`

### Categorical Features

Categorical variables were:

* Most-frequent imputed
* Encoded using `OneHotEncoder`

The preprocessing and model were combined using a **Pipeline** and **ColumnTransformer**.

---

## 🤖 Model Development

Two classification models were developed and compared:

### Logistic Regression

A Logistic Regression model was used as the primary linear classification model.

### Random Forest

A Random Forest classifier with **200 estimators** was also trained for comparison.

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score

Because the primary business objective is to identify customers at risk of churn, particular attention was given to the **Churn class recall and F1-score**.

---

## 📊 Model Comparison

Logistic Regression performed better than Random Forest on the evaluated metrics.

| Model               | Accuracy | Churn Precision | Churn Recall | Churn F1 |
| ------------------- | -------: | --------------: | -----------: | -------: |
| Logistic Regression |   0.8055 |            0.66 |         0.56 |     0.60 |
| Random Forest       |   0.7850 |            0.62 |         0.48 |     0.54 |

Based on this comparison, Logistic Regression was selected for further optimization.

---

## 🎛️ Hyperparameter Tuning

GridSearchCV was used to tune the Logistic Regression regularization parameter `C`.

Tested values:

`[0.01, 0.1, 1, 10, 100]`

The best parameter was:

**C = 1**

The best cross-validation F1-score was approximately:

**0.599**

The tuned Logistic Regression model achieved approximately:

**80.55% accuracy** on the test set.

---

## 📈 ROC-AUC Evaluation

The tuned model achieved an:

### **ROC-AUC = 0.842**

This indicates that the model has good ability to distinguish between customers who churn and customers who do not churn.

---

## 🎯 Threshold Optimization

The default classification threshold of **0.50** was evaluated against lower thresholds.

| Threshold | Precision |   Recall |       F1 |
| --------: | --------: | -------: | -------: |
|      0.30 |      0.52 | **0.76** | **0.62** |
|      0.35 |      0.54 |     0.71 |     0.61 |
|      0.40 |      0.57 |     0.67 |     0.61 |
|      0.45 |      0.60 |     0.62 |     0.61 |
|      0.50 |      0.66 |     0.56 |     0.60 |

A threshold of **0.30** was selected as the final threshold.

This increased churn recall from **56% to 76%**, allowing the model to identify more customers who actually churned.

The trade-off was lower precision, meaning more non-churning customers were also flagged as potential churners.

---

## 🔎 Model Interpretation

The final Logistic Regression model produced **46 transformed features** after preprocessing.

The strongest positive coefficients associated with churn included:

* Fiber optic internet service
* Month-to-month contracts
* Total Charges
* Electronic check payment
* Streaming TV
* Streaming Movies
* Lack of online security
* Lack of technical support

Strong negative coefficients included:

* Tenure
* Two-year contracts
* DSL internet service
* Monthly Charges
* Paperless billing set to No

The coefficient analysis was used to understand which transformed features were associated with higher or lower churn risk.

---

## 📌 Final Model Evaluation

At the selected threshold of **0.30**, the final model achieved:

* **Churn Precision:** 0.52
* **Churn Recall:** 0.76
* **Churn F1-score:** 0.62
* **Overall Accuracy:** 0.75

The final confusion matrix was:

```text
[[774 261]
 [ 91 283]]
```

This means the model correctly identified **283 of the 374 actual churners**.

The lower threshold prioritizes identifying more potential churners, which can be useful in a retention-focused business scenario.

---

## 💡 Business Insights

The analysis identified several important churn patterns:

1. **Month-to-month customers showed substantially higher churn risk** compared with customers on longer-term contracts.

2. **Fiber optic customers showed higher churn association** compared with the reference category.

3. **Shorter-tenure customers were more associated with churn**, while longer tenure was strongly associated with lower churn risk.

4. **Two-year contract customers showed substantially lower churn association.**

5. **Electronic check users showed the highest churn rate** among the payment methods analyzed.

---

## 🚀 Business Recommendations

Based on the analysis:

### 1. Encourage Longer-Term Contracts

Offer targeted incentives to month-to-month customers to encourage movement toward one-year or two-year contracts.

### 2. Focus on Newer Customers

Prioritize retention efforts for customers with shorter tenure, particularly during the early stages of the customer relationship.

### 3. Investigate Fiber Optic Churn

Investigate service quality, pricing, and customer support issues among fiber optic customers.

### 4. Promote Support & Security Services

Consider targeted support or security bundles for customers who do not currently use services such as online security and technical support.

### 5. Use the Model for Proactive Retention

Customers above the selected churn probability threshold can be prioritized for retention campaigns, allowing the business to focus resources on customers with higher predicted churn risk.

---

## 📁 Repository Contents

| File                                    | Description                          |
| --------------------------------------- | ------------------------------------ |
| `Telco_Customer_Churn_Prediction.ipynb` | Complete machine learning workflow   |
| `telco_churn_cleaned.csv`               | Cleaned dataset used by the notebook |
| `confusion_matrix.png`                  | Final model confusion matrix         |
| `roc_curve.png`                         | ROC curve and AUC visualization      |
| `README.md`                             | Project documentation                |

---

## 🧠 Key Takeaway

This project demonstrates how machine learning can be used not only to predict customer churn, but also to support business decision-making.

The final Logistic Regression model achieved an **ROC-AUC of 0.842**. More importantly, threshold optimization demonstrated how changing the classification threshold can prioritize **higher churn recall**, enabling the business to identify more potential churners for proactive retention.

The project combines **machine learning, model evaluation, interpretability, and business thinking** to turn customer data into actionable retention strategies.
