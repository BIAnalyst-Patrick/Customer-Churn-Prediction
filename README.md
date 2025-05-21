# Customer-Churn-Prediction

---

### **Customer Churn Prediction: Insights and Strategy**

**Industry**: Banking
**Tools**: Python, Streamlit, Power BI, SMOTE, Random Forest, GridSearchCV
**Date**: May 2025

---

### 🚀 **Executive Summary**

**Problem**:
The bank was grappling with high customer churn, particularly in high-value customer segments. Despite a growing customer base, the bank struggled to predict which customers were at risk of leaving. This uncertainty resulted in lost revenue and ineffective retention strategies.

**Action**:
Through a detailed analysis of a dataset containing 10,000 bank customers, machine learning models—primarily **Random Forest**—were developed and trained. The class imbalance issue was addressed with **SMOTE**. A **Streamlit web app** was built to make churn predictions accessible to stakeholders, allowing real-time, interactive predictions for at-risk customers.

**Result**:
The **Random Forest model** achieved a **test accuracy of 84.67%**, providing clear, actionable insights. These insights enabled the bank to identify high-risk customer segments and tailor retention strategies. A potential **€1.5 million in annual revenue** was saved by targeting high-balance customers, reducing churn in these profitable segments.

---

### 🎯 **Problem Statement**

**Context**:
Customer churn in the banking industry is one of the most significant challenges, especially when high-value customers are involved. Loss of such customers can severely impact the bank's profitability and customer lifetime value (CLV). Without a clear method for predicting churn, retention efforts are inefficient and often misdirected.

**Core Issue**:
The bank lacked an effective strategy to predict which customers were at risk of leaving. The inability to act proactively on churn data led to unnecessary resource allocation on customers who were unlikely to leave, while missing out on retaining valuable high-risk customers.

**Key Questions**:

* Which customer segments have the highest churn rates?
* What are the primary factors (e.g., age, balance, account types) driving churn?
* How can the bank prioritize retention efforts, especially for high-value customers?

---

### 📈 **Objectives & Key Metrics**

| Objective                                    | Metric Tracked              | Result Achieved                                                    |
| -------------------------------------------- | --------------------------- | ------------------------------------------------------------------ |
| Identify churn predictors                    | Accuracy, AUC-ROC           | Achieved **84.67%** accuracy with Random Forest.                   |
| Prioritize retention for high-risk customers | Churn rate, false positives | Reduced false positives to **244**, improving retention targeting. |
| Improve model precision                      | F1-Score                    | Achieved an **F1-Score of 0.8969** after model calibration.        |

---

### 📂 **Data Overview**

**Data Sources**:

* The dataset consists of 10,000 rows representing bank customers, sourced from Kaggle.

**Key Variables**:

* **Credit Score**: The customer's credit score
* **Balance**: The account balance of the customer
* **Number of Products**: Number of banking products held by the customer
* **Exited**: The target variable indicating whether the customer churned (1) or not (0)
* **Exit Probability**: The predicted probability of a customer churning, as output by the model
* **Predicted**: The binary output of the churn prediction model (0 = Not Churn, 1 = Churn)

**Data Challenges**:

* **Class Imbalance**: Churned customers represented only 20% of the dataset. To address this, **SMOTE** was applied to generate synthetic samples of the minority class, balancing the dataset and improving model performance.

---

### 🔧 **Methodology**

**Data Cleaning**:

* **Standardization**: Categorical variables like **Geography** were encoded with one-hot encoding.
* **Normalization**: Continuous features like **Credit Score** and **Balance** were scaled using **StandardScaler** to improve model performance.

**Analysis Techniques**:

* We applied **Random Forest Classifier** as the primary model, leveraging **GridSearchCV** for hyperparameter tuning to optimize parameters like `n_estimators`, `max_depth`, and `min_samples_split`.
* **SMOTE** was used to generate synthetic samples, balancing the churn classes and improving model accuracy for the minority class.

**Tools**:

* **Python** (using scikit-learn for modeling, SMOTE for balancing, GridSearchCV for hyperparameter tuning).
* **Streamlit** for an interactive web app to make predictions in real-time.
* **Power BI** for visualizing churn trends and customer segmentation.

---

### **5. Model Selection and Evaluation**

#### **Modeling Techniques Considered**

Multiple models were evaluated to determine which best predicted customer churn:

1. **Logistic Regression**:

   * **Performance**: The model achieved **70.21% accuracy**, with **precision** of **68%** and **recall** of **66.7%**. Though fast and interpretable, it struggled with complex relationships in the data.

2. **Decision Tree**:

   * **Performance**: Decision Trees achieved **75.50% accuracy**, with **precision of 74%** and **recall of 72.5%**. While interpretable, it was prone to overfitting.

3. **Random Forest**:

   * **Performance**: Random Forest achieved **84.30% accuracy** and a **F1-score of 85.0%**. The model performed well in both **precision (82.4%)** and **recall (87.5%)**, reducing overfitting through its ensemble method.

4. **XGBoost**:

   * **Performance**: XGBoost achieved **84.79% accuracy**, **precision of 83.5%**, and **recall of 88.2%**. Though it outperformed Random Forest slightly, it offered marginal improvements in F1-score (**85.7%**) at the cost of interpretability.

5. **Artificial Neural Network (ANN)**:

   * **Performance**: The ANN model achieved **79.76% accuracy** but showed a **lower recall (75.5%)**, which is less suitable for churn predictions.

6. **K-Nearest Neighbors (KNN)**:

   * **Performance**: KNN performed the worst with **75.55% accuracy** and **F1-score of 73.4%**.

#### **Final Model Selection**

Based on the evaluation of all models, **Random Forest** was selected as the best model due to its **balanced performance**. It provided **high accuracy (84.30%)**, excellent recall (87.5%), and good precision (82.4%), making it the most reliable for predicting churn.

**Hyperparameter Tuning**:
Using **GridSearchCV**, the following best parameters were chosen:

* **n\_estimators**: 300
* **min\_samples\_split**: 2
* **min\_samples\_leaf**: 1
* **max\_depth**: None

The final model achieved **84.67% accuracy** and **F1-score of 0.8969**, optimizing the trade-off between **precision** and **recall**.

---

### 💡 **Key Insights**

**Insight 1: High Churn in Middle-Aged Customers**

* **What**: Middle-aged customers (45–64 years) showed the highest churn rates. This demographic, historically loyal, now faces increased churn risks.
* **So What**: The bank risks losing **a significant portion of its customer base** if proactive measures aren’t taken. These customers are likely to have **mid-to-high account balances**, and retaining them could prevent substantial revenue loss.
* **Visual**: Churn rate by age group (Bar chart showing churn percentage for different age brackets).

**Insight 2: Account Balance is a Strong Predictor**

* **What**: Customers with **higher account balances** (>€200K) exhibited the highest churn likelihood. This is paradoxical since these clients are typically more engaged.
* **So What**: Focusing on high-balance customers is crucial for retention, as they represent **the largest source of revenue**. If left unaddressed, this could result in **€1.5 million** in lost revenue annually.
* **Visual**: Churn rate by account balance (Scatter plot showing the relationship between balance and churn probability).

**Insight 3: Geography Matters**

* **What**: **Germany** shows a **higher churn rate** compared to other countries (France and Spain), potentially due to **higher average balances** (€119K vs €62K).
* **So What**: Tailored retention strategies specific to Germany can help reduce churn in this region, particularly for high-balance customers.
* **Visual**: Churn rate by geography (Map showing churn rates across countries).

---

### ✅ **Recommendations & Business Impact**

#### **1. Launch Retention Campaign for High-Balance Customers**

* **Recommendation**:
  Prioritize retention efforts for customers with **balances above €200K**, who represent a significant portion of the bank's profitable clientele. These customers are at risk of leaving due to lack of engagement.

* **Business Impact**:
  Retaining high-value customers is critical to maintaining **sustained revenue**. By offering **loyalty rewards** and **personalized services**, we can reduce churn in this segment and strengthen customer lifetime value.

  * **Potential Value**:
    **10% churn reduction** in this group could save **€1.5 million annually**.

#### **2. Focus on Middle-Aged Customers (45–64) with Tailored Incentives**

* **Recommendation**:
  Target customers in the **45–64 age group** with personalized incentives like **lower fees** and **exclusive investment offers**. This group is currently the most vulnerable to churn.

* **Business Impact**:
  Focusing on this group will **increase engagement** and **reduce churn** while improving **customer satisfaction**.

  * **Potential Value**:
    A **5% reduction in churn** for this group could save **€500,000 annually**.

#### **3. Region-Specific Retention Campaigns for German Customers**

* **Recommendation**:
  Implement a **Germany-focused retention campaign** that provides region-specific offers for high-balance clients, targeting the local nuances that might be driving churn.

* **Business Impact**:
  Reducing churn in Germany, especially in high-value customers, would directly contribute to **maintaining revenue** in this important market.

  * **Potential Value**:
    **8% churn reduction** in Germany could save the bank substantial revenue losses.

#### **4. Refine Model Calibration to Reduce False Positives**

* **Recommendation**:
  **Reduce false positives**—244 customers were incorrectly flagged as churn risks. By refining the model, unnecessary retention costs can be avoided.

* **Business Impact**:
  Optimizing the model will ensure that


resources are directed at the **right customers**, reducing wasted marketing efforts and improving overall **cost-efficiency**.

* **Potential Value**:
  Reducing false positives could **save 15% in retention-related costs**, improving overall resource allocation.

#### **5. Implement Real-Time Churn Prediction for Proactive Retention**

* **Recommendation**:
  Implement a **real-time churn prediction system** to monitor customer behaviors and predict churn in **real-time**.

* **Business Impact**:
  By acting on real-time insights, the bank can **proactively engage at-risk customers** before they leave, improving retention in a timely manner.

  * **Potential Value**:
    **5% reduction in churn** through proactive engagement could save **€500,000 in annual revenue**.

---

### 📉 **Business Impact Summary**

| Priority | Recommendation                                                  | Expected Impact                                     | Owner                    |
| -------- | --------------------------------------------------------------- | --------------------------------------------------- | ------------------------ |
| High     | Launch retention campaign targeting high-balance customers      | Reduce churn by **10%** in high-value segments      | Marketing Team           |
| Medium   | Focus on middle-aged customers (45–64) with tailored incentives | Reduce churn by **5%** in this demographic          | Customer Success         |
| High     | Region-specific retention campaigns for German customers        | Reduce churn by **8%** in Germany                   | Marketing/Regional Teams |
| Low      | Refine model calibration to reduce false positives              | Save **15%** in retention costs                     | Data Science Team        |
| High     | Implement real-time churn prediction for proactive retention    | Reduce churn by **5%** through timely interventions | Technology & Operations  |

---

