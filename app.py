
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model & preprocessors
model = joblib.load("models/wine_quality_model.pkl")
imputer = joblib.load("models/imputer.pkl")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(
    page_title="Wine Quality Predictor", 
    layout="centered",
    page_icon="🍷"
)

st.title("🍷 Wine Quality Prediction")
st.markdown("---")
st.markdown("""
**Predict wine quality based on chemical properties**  
Enter the wine's chemical composition below to determine if it's **Good** or **Not Good** quality.
""")

with st.expander("ℹ️ Need help? Click here for typical ranges and example values"):
    st.markdown("""
    **Typical ranges for wine properties:**
    - **Fixed Acidity**: 4.6 - 15.9 g/L
    - **Volatile Acidity**: 0.12 - 1.58 g/L  
    - **Citric Acid**: 0.0 - 1.0 g/L
    - **Residual Sugar**: 0.9 - 15.5 g/L
    - **Chlorides**: 0.012 - 0.611 g/L
    - **Free Sulfur Dioxide**: 1 - 72 mg/L
    - **Total Sulfur Dioxide**: 6 - 289 mg/L
    - **Density**: 0.99 - 1.00 g/cm³
    - **pH**: 2.74 - 4.01
    - **Sulphates**: 0.33 - 2.0 g/L
    - **Alcohol**: 8.4 - 14.9% vol
    """)

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🍷 Load Good Wine Example"):
        st.session_state.example_loaded = "good"
with col2:
    if st.button("🍾 Load Average Wine Example"):
        st.session_state.example_loaded = "average"
with col3:
    if st.button("🔄 Clear All"):
        st.session_state.example_loaded = "clear"

good_wine_example = [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]
average_wine_example = [8.1, 0.28, 0.4, 6.9, 0.05, 30.0, 97.0, 0.9951, 3.26, 0.44, 10.1]

# Feature inputs with better organization
st.markdown("### 🧪 Wine Chemical Properties")

st.markdown("**Acidity Properties**")
col1, col2, col3 = st.columns(3)

feature_names = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]

inputs = []

default_values = [0.0] * 11
if hasattr(st.session_state, 'example_loaded'):
    if st.session_state.example_loaded == "good":
        default_values = good_wine_example
    elif st.session_state.example_loaded == "average":
        default_values = average_wine_example
    elif st.session_state.example_loaded == "clear":
        default_values = [0.0] * 11

with col1:
    inputs.append(st.number_input("Fixed Acidity (g/L)", value=default_values[0], format="%.3f", key="fixed_acidity"))
with col2:
    inputs.append(st.number_input("Volatile Acidity (g/L)", value=default_values[1], format="%.3f", key="volatile_acidity"))
with col3:
    inputs.append(st.number_input("Citric Acid (g/L)", value=default_values[2], format="%.3f", key="citric_acid"))

st.markdown("**Sugar and Minerals**")
col1, col2, col3 = st.columns(3)
with col1:
    inputs.append(st.number_input("Residual Sugar (g/L)", value=default_values[3], format="%.3f", key="residual_sugar"))
with col2:
    inputs.append(st.number_input("Chlorides (g/L)", value=default_values[4], format="%.3f", key="chlorides"))
with col3:
    inputs.append(st.number_input("Sulphates (g/L)", value=default_values[9], format="%.3f", key="sulphates"))

st.markdown("**Sulfur Dioxide**")
col1, col2 = st.columns(2)
with col1:
    inputs.insert(5, st.number_input("Free Sulfur Dioxide (mg/L)", value=default_values[5], format="%.1f", key="free_sulfur"))
with col2:
    inputs.insert(6, st.number_input("Total Sulfur Dioxide (mg/L)", value=default_values[6], format="%.1f", key="total_sulfur"))

st.markdown("**Physical Properties**")
col1, col2, col3 = st.columns(3)
with col1:
    inputs.insert(7, st.number_input("Density (g/cm³)", value=default_values[7], format="%.4f", key="density"))
with col2:
    inputs.insert(8, st.number_input("pH", value=default_values[8], format="%.2f", key="ph"))
with col3:
    inputs.append(st.number_input("Alcohol (% vol)", value=default_values[10], format="%.1f", key="alcohol"))

st.markdown("---")
st.markdown("### 🎯 Prediction")

valid_inputs = all(val > 0 for val in inputs[:5]) and all(val >= 0 for val in inputs[5:])

if not valid_inputs:
    st.warning("⚠️ Please enter valid values for all properties (most should be greater than 0)")

predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
with predict_col2:
    predict_button = st.button("🔮 Predict Wine Quality", disabled=not valid_inputs, use_container_width=True)

if predict_button and valid_inputs:
    # Convert input → DataFrame
    input_df = pd.DataFrame([inputs], columns=feature_names)

    # Apply same preprocessing
    X_imputed = imputer.transform(input_df)
    X_scaled = scaler.transform(X_imputed)

    # Prediction
    prediction = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0][1]

    st.markdown("### 📊 Results")
    
    if prediction == 1:
        st.success(f"✅ **Excellent!** This wine is predicted to be **Good Quality**")
        st.metric("Quality Probability", f"{proba:.1%}", delta="Good")
    else:
        st.error(f"❌ **Needs Improvement** This wine is predicted to be **Not Good Quality**")
        st.metric("Quality Probability", f"{(1-proba):.1%}", delta="Not Good")
    
    confidence = max(proba, 1-proba)
    if confidence > 0.8:
        st.info("🎯 **High Confidence** - The model is very confident in this prediction")
    elif confidence > 0.6:
        st.info("📊 **Medium Confidence** - The model has moderate confidence in this prediction")
    else:
        st.warning("🤔 **Low Confidence** - The model is uncertain about this prediction")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    <p>🍷 Wine Quality Predictor | Built with Streamlit</p>
    <p>This model analyzes 11 chemical properties to predict wine quality</p>
</div>
""", unsafe_allow_html=True)
