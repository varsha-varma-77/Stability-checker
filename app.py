import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

# Page Config
st.set_page_config(page_title="Emotional Stability AI", layout="wide")

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_stdio=True)

# --- SIDEBAR ---
st.sidebar.title("⚙️ Settings")
st.sidebar.info("Adjust your daily habits below to check your stability score.")

study = st.sidebar.slider("Study Hours", 0.0, 12.0, 7.0)
sleep = st.sidebar.slider("Sleep Hours", 0.0, 12.0, 8.0)
phys = st.sidebar.slider("Physical Activity", 0.0, 8.0, 2.0)
social = st.sidebar.slider("Social Hours", 0.0, 8.0, 3.0)
gpa = st.sidebar.number_input("Current GPA", 0.0, 4.0, 3.5)

# --- MAIN PANEL ---
st.title("🧠 Emotional Stability Checker")
st.divider()

# Load Data
@st.cache_data
def get_data():
    return pd.read_csv('student_lifestyle_dataset.csv')

try:
    df = get_data()
    
    # Train Model (Silent)
    X = df[['Study_Hours_Per_Day', 'Extracurricular_Hours_Per_Day', 'Sleep_Hours_Per_Day', 
            'Social_Hours_Per_Day', 'Physical_Activity_Hours_Per_Day', 'GPA']]
    y = df['Stress_Level']
    model = RandomForestClassifier().fit(X, y)

    # Top Row: Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Sleep (Dataset)", f"{df['Sleep_Hours_Per_Day'].mean():.1f}h")
    c2.metric("Avg GPA (Dataset)", f"{df['GPA'].mean():.2f}")
    c3.metric("Total Records", len(df))

    # Prediction Section
    st.subheader("🔍 Stability Prediction")
    if st.sidebar.button("RUN ANALYSIS"):
        # Note: Extracurricular is set to 2.0 as a default here
        pred = model.predict([[study, 2.0, sleep, social, phys, gpa]])[0]
        
        if pred == 'Low':
            st.success("### Status: HIGH STABILITY 🟢")
            st.balloons()
        elif pred == 'Moderate':
            st.warning("### Status: MODERATE STABILITY 🟡")
        else:
            st.error("### Status: LOW STABILITY (High Stress) 🔴")

    # Visuals
    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Stress Level Distribution**")
        fig1 = px.pie(df, names='Stress_Level', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig1, use_container_width=True)
    with col_b:
        st.write("**Study vs Sleep Correlation**")
        fig2 = px.scatter(df, x="Study_Hours_Per_Day", y="Sleep_Hours_Per_Day", color="Stress_Level")
        st.plotly_chart(fig2, use_container_width=True)

except Exception as e:
    st.error(f"Waiting for dataset... Ensure 'student_lifestyle_dataset.csv' is in your GitHub. Error: {e}")
