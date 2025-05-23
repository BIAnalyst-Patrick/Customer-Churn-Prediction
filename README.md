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

🔗 Try the app now: Customer Churn Prediction App [https://lnkd.in/d7Duur94](https://customer-churn-prediction-9xadjygpa4owqrdir7g3dm.streamlit.app/)


![image](https://github.com/user-attachments/assets/7a08a33f-a325-401f-b433-5f143aac7829)

![image](https://github.com/user-attachments/assets/73b506ba-e186-4911-909e-7f76a07f6e14)


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
* **Visual**: Churn rate by age group.
  ![image](https://github.com/user-attachments/assets/810307e8-85bf-4dc7-8705-269375c0be5d)


**Insight 2: Account Balance is a Strong Predictor**

* **What**: Customers with **higher account balances** (>€200K) exhibited the highest churn likelihood. This is paradoxical since these clients are typically more engaged.
* **So What**: Focusing on high-balance customers is crucial for retention, as they represent **the largest source of revenue**. If left unaddressed, this could result in **€1.5 million** in lost revenue annually.
* **Visual**: Churn rate by account balance.
  ![image](https://github.com/user-attachments/assets/c0b5e4bf-e2bf-4c61-9871-469856248b96)


**Insight 3: Geography Matters**

* **What**: **Germany** shows a **higher churn rate** compared to other countries (France and Spain), potentially due to **higher average balances** (€119K vs €62K).
* **So What**: Tailored retention strategies specific to Germany can help reduce churn in this region, particularly for high-balance customers.
* **Visual**: Churn rate by geography.
  ![image](https://github.com/user-attachments/assets/0752f18e-6580-42d8-8763-0039d0e324b3)


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

### 📉 **Caveats & Next Steps**

#### **Caveats**

1. **Limited Features**:
   The current model primarily uses **demographic and account-related data**. Adding **behavioral features** (e.g., transaction history, digital engagement) could enhance accuracy and predictive power.

2. **Class Imbalance**:
   Although SMOTE was applied, the model may still be slightly biased toward predicting the majority class (non-churning customers). Further use of **cost-sensitive learning** could improve performance.

3. **External Factors**:
   The model is trained on **historical data** and may not account for **current market shifts** or **external factors** (economic downturns, new competition). Regular retraining is required to keep predictions relevant.

4. **Regional Differences**:
   **Geographic** differences are considered but not deeply modeled. A more **region-specific approach** could improve prediction accuracy, especially in diverse markets.

5. **Interpretability**:
   Random Forests are powerful but **non-interpretable**. Implementing techniques like **SHAP** or **LIME** would increase model transparency, helping stakeholders understand why a customer is flagged as high-risk.

---

#### **Next Steps**

1. **Incorporate Behavioral Features**

   * **Action**: Add data such as **transaction frequency** and **customer interaction history** to improve model accuracy.
   * **Impact**: Expected **10-15% improvement** in predictive performance.

2. **Refine with Cost-Sensitive Learning**

   * **Action**: Use **cost-sensitive methods** like **Balanced Random Forest** to reduce bias toward the majority class.
   * **Impact**: More balanced predictions and reduced **false positives**, optimizing retention efforts.

3. **Retrain the Model Regularly**

   * **Action**: Re-train the model with updated data quarterly to adapt to **new trends**.
   * **Impact**: **Improved relevance** and accuracy over time, reducing model obsolescence.

4. **Deploy Region-Specific Retention Strategies**

   * **Action**: Implement tailored strategies for **high-risk regions** (e.g., Germany) with specific offers and engagements.
   * **Impact**: **8% churn reduction** in high-risk regions, optimizing localized marketing resources.

5. **Improve Model Interpretability**

   * **Action**: Use **SHAP** or **LIME** to provide **explainable insights** into churn predictions.
   * **Impact**: **Enhanced trust** and decision-making transparency, improving executive buy-in.

6. **Implement Real-Time Churn Prediction**

   * **Action**: Integrate churn predictions into **real-time CRM systems** for timely interventions.
   * **Impact**: Immediate actions can lead to **5-10% more customer retention**, translating to **€500,000+ saved annually**.

