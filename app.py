
categorical_cols = ['contract_type', 'payment_method']
numerical_cols = ['tenure_months', 'monthly_charges', 'total_charges', 'tech_support_calls']
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------------------
st.set_page_config(page_title="AI Retention Agent", layout="wide")

st.title("🤖 AI-Driven Customer Retention Agent")
st.markdown("""
**Role:** Senior Data Scientist & AI Engineer.  
**Goal:** Predict customer churn, explain *why* it's happening, and draft a personalized retention email using GenAI logic.
""")

# ------------------------------------------------------------------------
# 2. SYNTHETIC DATA GENERATION
# ------------------------------------------------------------------------
@st.cache_data
def generate_data():
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'customer_id': range(1000, 1000 + n_samples),
        'tenure_months': np.random.randint(1, 72, n_samples),
        'monthly_charges': np.random.uniform(30, 120, n_samples),
        'total_charges': np.random.uniform(100, 8000, n_samples),
        'tech_support_calls': np.random.randint(0, 5, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Credit card'], n_samples)
    })
    
    # Simulate Churn Logic (High charges + Low tenure = Higher Churn Risk)
    data['churn_prob'] = (
        (data['monthly_charges'] / 120) * 0.4 +
        (1 - data['tenure_months'] / 72) * 0.4 + 
        (data['tech_support_calls'] / 5) * 0.2
    )
    # Add some noise
    data['churn_prob'] += np.random.normal(0, 0.1, n_samples)
    data['churn'] = (data['churn_prob'] > 0.65).astype(int)
    
    return data.drop(columns=['churn_prob'])

df = generate_data()

# ------------------------------------------------------------------------
# 3. MACHINE LEARNING PIPELINE
# ------------------------------------------------------------------------
@st.cache_resource
def train_model(data):
    X = data.drop(columns=['churn', 'customer_id'])
    y = data['churn']
    
    # Define categorical and numerical features
    categorical_cols = ['contract_type', 'payment_method']
    numerical_cols = ['tenure_months', 'monthly_charges', 'total_charges', 'tech_support_calls']
    
    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    # Pipeline with SMOTE (Handling Imbalance) and XGBoost
    model_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit Model
    model_pipeline.fit(X_train, y_train)
    
    # Metrics
    y_pred = model_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    
    return model_pipeline, X_test, y_test, acc, auc

model, X_test, y_test, acc, auc = train_model(df)

# Sidebar Metrics
st.sidebar.header("Model Performance")
st.sidebar.metric("Accuracy", f"{acc:.2%}")
st.sidebar.metric("ROC-AUC Score", f"{auc:.2f}")
st.sidebar.markdown("---")

# ------------------------------------------------------------------------
# 4. DASHBOARD UI
# ------------------------------------------------------------------------

# Section 1: Data Preview
with st.expander("📊 View Raw Data (First 5 Rows)", expanded=False):
    st.dataframe(df.head())

# Section 2: Customer Selection
st.subheader("🔎 Customer Analysis")
selected_customer_idx = st.selectbox("Select a Customer to Analyze (Test Set):", X_test.index)
customer_data = X_test.loc[[selected_customer_idx]]

# Inference
churn_prob = model.predict_proba(customer_data)[0][1]
churn_prediction = "High Risk" if churn_prob > 0.5 else "Low Risk"

col1, col2 = st.columns(2)
with col1:
    st.info(f"**Customer ID:** {df.loc[selected_customer_idx, 'customer_id']}")
with col2:
    if churn_prob > 0.5:
        st.error(f"**Prediction:** {churn_prediction} ({churn_prob:.1%})")
    else:
        st.success(f"**Prediction:** {churn_prediction} ({churn_prob:.1%})")

# ------------------------------------------------------------------------
# 5. EXPLAINABILITY (SHAP)
# ------------------------------------------------------------------------
st.subheader("💡 Why is this customer at risk?")

# Prepare data for SHAP (Transforming the single row using the preprocessor)
# Note: SHAP with pipelines can be tricky. We access the preprocessor step directly.
preprocessor = model.named_steps['preprocessor']
X_transformed = preprocessor.transform(customer_data)
feature_names = (numerical_cols + 
                 list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)))

# Create SHAP Explainer (Using the XGBoost model inside the pipeline)
explainer = shap.TreeExplainer(model.named_steps['classifier'])
shap_values = explainer.shap_values(X_transformed)


# NEW CODE (Fixes the yellow box)
import matplotlib.pyplot as plt # Make sure this import is here

shap.force_plot(explainer.expected_value, shap_values[0], feature_names=feature_names, matplotlib=True)
st.pyplot(plt.gcf()) # Explicitly grabs the current figure instead

shap_data = pd.DataFrame({
    'feature': feature_names,
    'shap_value': shap_values[0]
}).sort_values(by='shap_value', ascending=True) # Ascending because negative SHAP pushes towards class 0 (stay), positive towards 1 (churn)

# For Churn, we care about Positive SHAP values (pushing towards Churn)
top_factors = shap_data.sort_values(by='shap_value', ascending=False).head(2)
factor_1 = top_factors.iloc[0]['feature']
factor_2 = top_factors.iloc[1]['feature']

# ------------------------------------------------------------------------
# 6. GENERATIVE AI AGENT (MOCKED)
# ------------------------------------------------------------------------
st.subheader("📧 AI-Drafted Retention Email")
st.markdown("The system uses **LangChain** logic to inject the specific risk factors above into a personalized email.")

# Define the Prompt Template
email_template = """
**Subject:** Let's discuss your experience with {contract_type} plan

Dear Customer,

I noticed you've been with us for {tenure} months. We truly value your loyalty.

However, our system flagged some recent friction points, specifically regarding **{factor_1}** and **{factor_2}**. 
We want to fix this.

I've authorized a **15% discount** for your next 3 months to make up for the trouble. 
Are you free for a 5-minute chat tomorrow?

Best,
The Customer Success AI
"""

generated_email = email_template.format(
    tenure=int(customer_data['tenure_months'].values[0]),
    contract_type=customer_data['contract_type'].values[0],
    factor_1=factor_1.replace('num__', '').replace('cat__', '').replace('_', ' ').title(),
    factor_2=factor_2.replace('num__', '').replace('cat__', '').replace('_', ' ').title()
)

st.text_area("Generated Output:", value=generated_email, height=300)