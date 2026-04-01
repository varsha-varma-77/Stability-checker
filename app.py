import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Page configuration
st.set_page_config(page_title="Emotional Stability Checker", page_icon="🧠", layout="centered")

st.title("🧠 Emotional Stability Checker")
st.markdown("""
    This tool analyzes your daily lifestyle habits to assess your current emotional stability 
    and stress patterns based on data from 2,000 students.
""")

# --- STEP 1: LOAD DATA & TRAIN MODEL ---
@st.cache_data
def load_and_train():
    df = pd.read_csv('student_lifestyle_dataset.csv')
    X = df.drop(columns=['Student_ID', 'Stress_Level'])
    y = df['Stress_Level']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = load_and_train()

# --- STEP 2: USER INPUT SECTION ---
st.subheader("Input Your Daily Habits")
col1, col2 = st.columns(2)

with col1:
    study = st.slider("Study Hours", 0.0, 12.0, 7.0)
    sleep = st.slider("Sleep Hours", 0.0, 12.0, 8.0)
    extra = st.slider("Extracurricular Hours", 0.0, 5.0, 2.0)

with col2:
    social = st.slider("Social Hours", 0.0, 10.0, 3.0)
    physical = st.slider("Physical Activity (Hours)", 0.0, 10.0, 1.0)
    gpa = st.number_input("Current GPA", 0.0, 4.0, 3.5)

# --- STEP 3: PREDICTION LOGIC ---
if st.button("Check My Stability"):
    # Create input row for the model
    user_data = pd.DataFrame([[study, extra, sleep, social, physical, gpa]], 
                             columns=['Study_Hours_Per_Day', 'Extracurricular_Hours_Per_Day', 
                                      'Sleep_Hours_Per_Day', 'Social_Hours_Per_Day', 
                                      'Physical_Activity_Hours_Per_Day', 'GPA'])
    
    prediction = model.predict(user_data)[0]
    
    # Mapping Stress Level to Emotional Stability
    if prediction == 'Low':
        st.success("### Result: High Emotional Stability 🌟")
        st.write("Your habits suggest a very stable and low-stress lifestyle. Keep it up!")
    elif prediction == 'Moderate':
        st.warning("### Result: Balanced/Moderate Stability ⚖️")
        st.write("You are maintaining a decent balance, but be careful of increasing study loads.")
    else:
        st.error("### Result: Low Emotional Stability (Action Needed) ⚠️")
        st.write("Your current habits indicate high stress. Try increasing your sleep or physical activity.")

st.sidebar.info("This AI model was trained on the Student Lifestyle Dataset.")
