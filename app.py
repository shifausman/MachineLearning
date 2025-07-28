import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ---- Sample dataset ----
data = {
    "Year": [2019, 2020, 2021, 2022, 2023],
    "SM": [2194, 1965, 1686, 2039, 2660],
    "EZ": [2843, 2218, 2000, 2847, 3381],
    "MU": [2227, 2151, 1822, 2526, 3086],
    "LA": [3315, 4956, 2994, 5180, 4754],
    "DV": [5628, 5731, 6287, 7884, 5674],
    "VK": [3320, 2426, 2521, 2452, 4247],
    "BH": [2904, 2916, 2516, 2343, 3184],
    "BX": [4871, 8703, 8305, 7804, 8565],
    "KN": [42082, 18934, 17350, 15809, 16224],
    "KU": [22809, 7446, 7000, 6065, 10352],
    "SC": [26000, 25542, 20356, 19265, 20516],
    "ST": [13651, 41394, 34619, 54300, 14003],
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
