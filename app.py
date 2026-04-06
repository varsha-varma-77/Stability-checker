import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Emotional Stability Checker", page_icon="🌱", layout="centered")

# ── THEME CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,300;0,400;0,600;0,700;1,400&display=swap');
html, body, [class*="css"] { background-color:#0f1a14!important; color:#e8f5ec!important; font-family:'Crimson Pro',Georgia,serif!important; }
.block-container { padding-top:2rem; max-width:680px; }
h1,h2,h3 { color:#e8f5ec!important; }
div[data-testid="metric-container"] { background:#16261c; border:1px solid #2d4a38; border-radius:14px; padding:16px 20px!important; }
div[data-testid="metric-container"] label { color:#7aab8a!important; font-size:12px!important; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color:#4ade80!important; font-size:26px!important; font-weight:700!important; }
.stSlider>div>div>div { background:#4ade80!important; }
.stSelectbox>div>div { background:#16261c!important; border-color:#2d4a38!important; color:#e8f5ec!important; }
.stButton>button { background:#4ade80!important; color:#0f1a14!important; border:none!important; border-radius:50px!important; font-weight:700!important; font-size:16px!important; padding:10px 40px!important; font-family:'Crimson Pro',Georgia,serif!important; width:100%; }
.stButton>button:hover { background:#86efac!important; }
section[data-testid="stSidebar"] { background:#16261c!important; border-right:1px solid #2d4a38; }
section[data-testid="stSidebar"] * { color:#e8f5ec!important; }
section[data-testid="stSidebar"] .stSelectbox>div>div { background:#0f1a14!important; }
details { border:1px solid #2d4a38!important; border-radius:12px!important; background:#16261c!important; }
details summary { color:#7aab8a!important; }
hr { border-color:#2d4a38!important; }
p, li, label { color:#e8f5ec!important; }
.card { background:#16261c; border:1px solid #2d4a38; border-radius:16px; padding:22px 26px; margin:14px 0; }
button[data-baseweb="tab"] { color:#7aab8a!important; background:transparent!important; }
button[data-baseweb="tab"][aria-selected="true"] { color:#4ade80!important; border-bottom:2px solid #4ade80!important; }
div[data-baseweb="tab-list"] { background:transparent!important; border-bottom:1px solid #2d4a38!important; }
</style>
""", unsafe_allow_html=True)

# ── COLOURS ──────────────────────────────────────────────────────────────────
BG, CARD, ACCENT, MUTED = "#0f1a14", "#16261c", "#4ade80", "#2d4a38"
TEXT, DIM, WARN, DANGER  = "#e8f5ec", "#7aab8a", "#fb923c", "#f87171"

def sc(lbl): return {" Low": ACCENT, "Low": ACCENT, "Moderate": WARN, "High": DANGER}.get(lbl, ACCENT)
def se(lbl): return {"Low": "🌟", "Moderate": "⚡", "High": "🆘"}.get(lbl, "🌱")

def dark_fig(*args, **kwargs):
    fig, ax = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(CARD)
    axes = [ax] if not hasattr(ax, '__iter__') else ax.flatten()
    for a in axes:
        a.set_facecolor(CARD)
        a.tick_params(colors=DIM, labelsize=10)
        a.xaxis.label.set_color(DIM)
        a.yaxis.label.set_color(DIM)
        a.title.set_color(TEXT)
        for s in a.spines.values(): s.set_edgecolor(MUTED)
    return fig, ax

# ── LOAD & TRAIN (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🌱 Training Random Forest…")
def train():
    df = pd.read_csv("student_lifestyle_dataset.csv")
    FEATS = ["Study_Hours_Per_Day","Extracurricular_Hours_Per_Day",
             "Sleep_Hours_Per_Day","Social_Hours_Per_Day",
             "Physical_Activity_Hours_Per_Day","GPA"]
    X = df[FEATS]
    le = LabelEncoder()
    y  = le.fit_transform(df["Stress_Level"])
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    rf = RandomForestClassifier(n_estimators=300,max_depth=10,random_state=42,n_jobs=-1)
    rf.fit(Xtr,ytr)
    yp = rf.predict(Xte)
    imp = pd.Series(rf.feature_importances_, index=FEATS).sort_values()
    rpt = classification_report(yte,yp,target_names=le.classes_,output_dict=True)
    return rf, le, accuracy_score(yte,yp), confusion_matrix(yte,yp), rpt, imp, FEATS, df

rf, le, acc, cm, rpt, imp, FEATS, df = train()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌱 Model Info")
    st.markdown(f"""
<div class='card' style='padding:14px 18px;'>
<b>Algorithm:</b> Random Forest<br>
<b>Estimators:</b> 300 trees<br>
<b>Dataset:</b> 2,000 students<br>
<b>Features:</b> 6 lifestyle metrics<br>
<b>Target:</b> Stress Level
</div>""", unsafe_allow_html=True)

    st.metric("🎯 Accuracy", f"{acc*100:.1f}%")

    with st.expander("📊 Per-class metrics"):
        for cls in ["Low","Moderate","High"]:
            col1,col2,col3 = st.columns(3)
            col1.metric(cls,"")
            col2.metric("Prec", f"{rpt[cls]['precision']:.2f}")
            col3.metric("Rec",  f"{rpt[cls]['recall']:.2f}")

    st.markdown("---")
    st.caption("For self-reflection only. Not medical advice.")

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:16px 0 4px 0;'>
  <div style='font-size:58px;'>🌱</div>
  <h1 style='font-size:38px;font-weight:300;line-height:1.2;margin:6px 0 4px 0;'>
    Emotional <em style='font-weight:700;color:#4ade80;'>Stability</em> Checker
  </h1>
  <p style='color:#7aab8a;font-size:15px;'>
    Powered by Random Forest · Trained on 2,000 students
  </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🧠  Check My Stability", "📊  Model Insights"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — USER ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Tell us about your daily life")
    st.caption("Adjust the sliders to reflect your average day.")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**🎓 Academic**")
    study = st.slider("Study Hours / Day",       0.0, 14.0, 6.0, 0.1)
    gpa   = st.slider("GPA (out of 4.0)",        2.0,  4.0, 3.1, 0.01)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**🌿 Lifestyle**")
    sleep  = st.slider("Sleep Hours / Day",           4.0, 12.0, 7.0, 0.1)
    phys   = st.slider("Physical Activity Hours / Day",0.0,  6.0, 1.0, 0.1)
    social = st.slider("Social Hours / Day",           0.0,  8.0, 2.0, 0.1)
    extra  = st.slider("Extracurricular Hours / Day",  0.0,  6.0, 1.0, 0.1)
    st.markdown("</div>", unsafe_allow_html=True)

    predict_btn = st.button("✦  Analyse My Stability")

    if predict_btn:
        inp = pd.DataFrame([[study, extra, sleep, social, phys, gpa]], columns=FEATS)
        pred_idx  = rf.predict(inp)[0]
        pred_prob = rf.predict_proba(inp)[0]
        pred_lbl  = le.inverse_transform([pred_idx])[0]
        color     = sc(pred_lbl)
        emoji     = se(pred_lbl)

        st.divider()

        # ── RESULT HERO ──
        st.markdown(f"""
<div style='text-align:center;padding:28px 0 16px 0;'>
  <div style='font-size:64px;'>{emoji}</div>
  <div style='font-size:34px;font-weight:700;color:{color};margin:8px 0 4px 0;'>{pred_lbl} Stress</div>
  <div style='color:#7aab8a;font-size:15px;'>Predicted Stress Level</div>
</div>
""", unsafe_allow_html=True)

        # ── PROBABILITY BARS ──
        st.markdown("#### Confidence Breakdown")
        classes = le.classes_
        for i, (cls, prob) in enumerate(zip(classes, pred_prob)):
            bar_color = sc(cls)
            st.markdown(f"<div style='display:flex;justify-content:space-between;font-size:14px;margin-bottom:4px;'>"
                        f"<span>{se(cls)} {cls}</span><span style='color:{bar_color};font-weight:700;'>{prob*100:.1f}%</span></div>",
                        unsafe_allow_html=True)
            st.progress(float(prob))

        st.divider()

        # ── RADAR / PROFILE CHART ──
        st.markdown("#### Your Lifestyle Profile")
        labels  = ["Study", "Sleep", "Physical\nActivity", "Social", "Extracurr.", "GPA×2.5"]
        values  = [study, sleep, phys, social, extra, gpa * 2.5]
        N       = len(labels)
        angles  = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        values_plot = values + [values[0]]
        angles_plot = angles + [angles[0]]

        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor(CARD)
        ax.set_facecolor(CARD)
        ax.plot(angles_plot, values_plot, color=ACCENT, linewidth=2.5)
        ax.fill(angles_plot, values_plot, color=ACCENT, alpha=0.20)
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, color=DIM, fontsize=10)
        ax.tick_params(colors=DIM)
        ax.yaxis.set_tick_params(labelcolor=DIM, labelsize=8)
        ax.spines['polar'].set_color(MUTED)
        ax.grid(color=MUTED, linestyle='--', linewidth=0.6)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # ── PERSONALISED TIPS ──
        st.divider()
        st.markdown("#### 💡 Personalised Suggestions")
        tips = []
        if sleep < 6:       tips.append(("😴", "You're sleeping under 6 hours — aim for 7–9 hrs. Sleep is the #1 recovery tool for stress."))
        if phys < 0.5:      tips.append(("🏃", "Even 20 mins of walking daily releases endorphins and measurably reduces stress hormones."))
        if study > 9:       tips.append(("📚", "Heavy study load detected. Use the Pomodoro method and schedule breaks to avoid burnout."))
        if social < 1:      tips.append(("💬", "Low social time. Reach out to one friend this week — connection is a stress buffer."))
        if gpa < 2.7:       tips.append(("🎓", "Academic pressure can spike stress. Talk to your academic advisor for support strategies."))
        if extra < 0.5:     tips.append(("🎨", "Try joining one extracurricular — hobbies and clubs act as mental recharge zones."))
        if pred_lbl == "High": tips.append(("🩺", "High stress flagged. Your campus counselling centre offers free, confidential support."))
        if not tips:        tips.append(("✨", "Great balance! Keep journaling to maintain self-awareness and protect your wellbeing."))

        for icon, tip in tips[:4]:
            st.markdown(f"""
<div style='display:flex;gap:14px;align-items:flex-start;background:#2d4a3866;
     border-radius:12px;padding:12px 16px;margin-bottom:10px;'>
  <span style='font-size:22px;'>{icon}</span>
  <p style='margin:0;font-size:14px;line-height:1.6;color:#e8f5ec;'>{tip}</p>
</div>""", unsafe_allow_html=True)

        st.caption("⚠️ This tool is for self-reflection only and does not constitute medical advice.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Model Performance & Dataset Insights")

    c1, c2, c3 = st.columns(3)
    c1.metric("🎯 Accuracy",   f"{acc*100:.1f}%")
    c2.metric("🌲 Trees",      "300")
    c3.metric("📦 Samples",    "2,000")

    st.divider()

    # ── FEATURE IMPORTANCE ──
    st.markdown("#### 🔍 Feature Importance")
    fig, ax = dark_fig(figsize=(7, 3.5))
    bars = ax.barh(imp.index, imp.values, color=ACCENT, edgecolor=MUTED, height=0.6)
    for bar, val in zip(bars, imp.values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va='center', color=DIM, fontsize=9)
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance (Random Forest)", pad=10)
    ax.tick_params(axis='y', labelsize=10)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── CONFUSION MATRIX ──
    st.markdown("#### 🧮 Confusion Matrix")
    fig, ax = dark_fig(figsize=(5, 4))
    classes = le.classes_
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, color=DIM)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, color=DIM)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix", pad=10)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color=TEXT if cm[i,j] < cm.max()/1.5 else BG, fontsize=13, fontweight='bold')
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04).ax.tick_params(colors=DIM)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── DATASET DISTRIBUTIONS ──
    st.markdown("#### 📈 Dataset Distributions")
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    fig.patch.set_facecolor(CARD)
    colors_map = {"Low": ACCENT, "Moderate": WARN, "High": DANGER}

    for ax_i, feat in zip(axes.flatten(), FEATS):
        ax_i.set_facecolor(CARD)
        for sl in df["Stress_Level"].unique():
            subset = df[df["Stress_Level"] == sl][feat]
            ax_i.hist(subset, bins=20, alpha=0.65, color=colors_map[sl], label=sl, edgecolor='none')
        ax_i.set_title(feat.replace("_"," ").replace("Per Day","/ Day"), color=TEXT, fontsize=10, pad=6)
        ax_i.tick_params(colors=DIM, labelsize=8)
        for s in ax_i.spines.values(): s.set_edgecolor(MUTED)

    patches = [mpatches.Patch(color=c, label=l) for l, c in colors_map.items()]
    fig.legend(handles=patches, loc='lower center', ncol=3, frameon=False,
               labelcolor=TEXT, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── STRESS DISTRIBUTION PIE ──
    st.markdown("#### 🥧 Stress Level Distribution in Dataset")
    counts = df["Stress_Level"].value_counts()
    fig, ax = dark_fig(figsize=(5, 4))
    wedge_colors = [colors_map[c] for c in counts.index]
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index, colors=wedge_colors,
        autopct="%1.1f%%", startangle=140, pctdistance=0.75,
        wedgeprops=dict(edgecolor=CARD, linewidth=2)
    )
    for t in texts: t.set_color(DIM)
    for at in autotexts: at.set_color(BG); at.set_fontweight('bold')
    ax.set_title("Stress Level Split", color=TEXT, pad=10)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption("Random Forest trained with sklearn · 80/20 train-test split · stratified sampling")
