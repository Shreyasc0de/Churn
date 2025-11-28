# 🤖 AI-Driven Customer Retention Agent

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Model-orange)](https://xgboost.readthedocs.io/)
[![LangChain](https://img.shields.io/badge/LangChain-GenAI-green)](https://www.langchain.com/)

An end-to-end Machine Learning application that predicts customer churn, explains the root causes using **SHAP (Explainable AI)**, and automatically drafts personalized retention emails using **Generative AI**.

![Dashboard Preview](https://raw.githubusercontent.com/Shreyasc0de/Churn/main/dashboard_screenshot_1.png)

---

## 💼 Business Value
Customer churn is a multi-million dollar problem for telecom companies. Traditional dashboards tell you *who* is leaving, but they don't tell you *why* or *what to do about it*. 

This application bridges that gap by:
1.  **Predicting Risk:** identifying high-risk customers with **83% accuracy** using XGBoost.
2.  **Explaining Decisions:** Using **SHAP values** to show exactly which factors (e.g., "High Monthly Charges") drove the prediction.
3.  **Automating Action:** Using **LLM Agents (LangChain)** to draft hyper-personalized emails that address those specific pain points.

---

## 🌟 Key Features

### 1. 🧠 Advanced ML Pipeline
* **Imbalance Handling:** Utilized **SMOTE** (Synthetic Minority Over-sampling Technique) to train on a balanced dataset.
* **Model:** XGBoost Classifier optimized for recall (minimizing missed churners).
* **Preprocessing:** Scikit-Learn `ColumnTransformer` for seamless numeric scaling and categorical encoding.

### 2. 🔍 Explainable AI (XAI)
* Integrated **SHAP (SHapley Additive exPlanations)** to provide local interpretability.
* Visualizes force plots to show how each feature (e.g., tenure, contract type) pushes the risk score up or down.

### 3. 📧 Generative AI Agent
* Built a **LangChain Prompt Template** system.
* Dynamically injects customer-specific SHAP insights into the prompt.
* Generates empathetic, context-aware emails offering discounts based on the customer's actual grievances.

---

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Frontend** | Streamlit | Interactive web dashboard for live inference |
| **Model** | XGBoost | Gradient boosting framework for classification |
| **Explainability** | SHAP | Game-theoretic approach to explain model output |
| **Orchestration** | LangChain | Framework for LLM-driven applications |
| **Data Processing** | Pandas / NumPy | Feature engineering and manipulation |
| **Handling Imbalance** | Imbalanced-Learn | SMOTE implementation |

---

## 🚀 How to Run Locally

**1. Clone the Repository**
```bash
git clone https://github.com/Shreyasc0de/Churn.git
cd Churn
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```

**3. Launch the App**
```bash
streamlit run app.py
```

---

## 📊 Dashboard Preview

![Dashboard Preview 1](https://raw.githubusercontent.com/Shreyasc0de/Churn/main/dashboard_screenshot_1.png)
![Dashboard Preview 2](https://raw.githubusercontent.com/Shreyasc0de/Churn/main/dashboard_screenshot_2.png)

---

## 📄 License
MIT
