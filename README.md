# 🌱 Emotional Stability Checker

A Streamlit web app that predicts a student's **stress level** (Low / Moderate / High) using a **Random Forest classifier** trained on real student lifestyle data.

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/emotional-stability-checker.git
cd emotional-stability-checker

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

> Make sure `student_lifestyle_dataset.csv` is in the **same folder** as `app.py`.

---

## 📦 Deploy on Streamlit Cloud (free)

1. Push this repo to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) → **New app**
3. Select your repo, set `app.py` as the main file
4. Click **Deploy** — done ✅

---

## 📊 Features

| Tab | What it shows |
|-----|--------------|
| 🧠 Check My Stability | Sliders → Random Forest prediction → confidence bars + radar chart + personalised tips |
| 📊 Model Insights | Feature importance · Confusion matrix · Dataset distributions · Stress split pie chart |

---

## 🗂 Dataset Columns

| Column | Description |
|--------|-------------|
| `Study_Hours_Per_Day` | Daily study hours |
| `Extracurricular_Hours_Per_Day` | Extracurricular time |
| `Sleep_Hours_Per_Day` | Sleep hours |
| `Social_Hours_Per_Day` | Social time |
| `Physical_Activity_Hours_Per_Day` | Exercise time |
| `GPA` | Grade Point Average (0–4) |
| `Stress_Level` | **Target**: Low / Moderate / High |

---

## 🛠 Tech Stack

- **Python 3.10+**
- **Streamlit** — UI
- **scikit-learn** — Random Forest
- **Matplotlib** — Charts
- **Pandas / NumPy** — Data processing

---

> ⚠️ This tool is for self-reflection only and does not constitute medical advice.
