import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ---- Sample dataset ----
data = {
    "Year": [2021, 2022, 2023],
    "GEN": [5400, 5231, 5123],
    "MU": [8600, 8234, 8012],
    "OBH": [7800, 7501, 7345],
    "LC": [7200, 6950, 6789],
    "SC": [14800, 14000, 13654],
    "ST": [25300, 24700, 24012],
}

df = pd.DataFrame(data)

# ---- Train models for each category ----
models = {}
categories = df.columns[1:]

for cat in categories:
    X = df[["Year"]]
    y = df[cat]
    model = LinearRegression()
    model.fit(X, y)
    models[cat] = model

# ---- Streamlit UI ----
st.set_page_config(page_title="KEAM CSE Admission Predictor", layout="centered")

st.title("🎓 KEAM CSE Admission Predictor - MACE")
st.markdown("#### 📌 Enter your KEAM rank and category to check your admission possibility at **Mar Athanasius College of Engineering** (CSE).")

rank = st.number_input("📥 Enter your KEAM 2025 Rank", min_value=1, value=5000)
category = st.selectbox("📂 Select Your Category", list(categories))

if st.button("🔍 Check Possibility"):
    predicted_cutoff = models[category].predict([[2025]])[0]
    st.markdown(f"### 📊 Predicted Last Rank for {category} in 2025: **{int(predicted_cutoff)}**")
    
    if rank <= predicted_cutoff:
        st.success("✅ You have a **high chance** of getting admission!")
    else:
        st.error("❌ You might **not get admission**, consider backup options.")

# ---- Optional: Display trend graph ----
import matplotlib.pyplot as plt

st.markdown("### 📈 Category-wise Rank Trend (2021–2025 Prediction)")
fig, ax = plt.subplots(figsize=(8, 5))

for cat in categories:
    years = list(df["Year"]) + [2024, 2025]
    preds = list(df[cat]) + list(models[cat].predict([[2024], [2025]]))
    ax.plot(years, preds, label=cat)

ax.set_xlabel("Year")
ax.set_ylabel("Last Rank")
ax.set_title("KEAM CSE Last Rank Trends by Category")
ax.legend()
st.pyplot(fig)
