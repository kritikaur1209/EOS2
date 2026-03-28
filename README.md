# 🧠 EmotionOS — Founder's Intelligence Dashboard

> **AI-powered market research analytics for EmotionOS — The Emotional Intelligence Layer for Global Brands**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-ff4b4b.svg)](https://streamlit.io)

---

## 🚀 Live App

Deploy to Streamlit Community Cloud — see [Deployment](#-deployment) below.

---

## 📌 What This Dashboard Does

End-to-end data-driven analysis of the EmotionOS market research survey (2,000 global respondents):

| Tab | Analysis Type | Methods |
|-----|--------------|---------|
| 🏠 Home | KPI Overview | Metrics, charts |
| 📊 Descriptive | Market composition, feature demand, budget | EDA, heatmaps, treemaps |
| 🔍 Diagnostic | Adoption drivers, barriers, segment compare | Correlation, drill-down, stacked bar |
| 🤖 Predictive | Adoption classification + budget regression | Random Forest, Gradient Boosting, Logistic Regression, Linear Regression |
| 🧩 Clustering | Customer tribe discovery | K-Means++, PCA, Elbow, Silhouette |
| 🔗 Association Rules | Feature bundles, pain→feature, source→channel | Apriori (mlxtend) |
| 🚀 New Client Predictor | Score any new brand → strategy card | All trained models + cluster assignment |

---

## 📁 File Structure (flat — no sub-folders)

```
app.py                          ← Main entry point
data_loader.py                  ← CSV loading + ML preprocessing
tab_home.py                     ← Home KPI tab
tab_descriptive.py              ← Descriptive analysis
tab_diagnostic.py               ← Diagnostic analysis
tab_predictive.py               ← Classification + Regression
tab_clustering.py               ← K-Means clustering
tab_arm.py                      ← Association Rule Mining
tab_predictor.py                ← New Client Predictor
EmotionOS_Survey_Dataset.csv    ← 2,000-row synthetic survey data
EmotionOS_Data_Dictionary.csv   ← Column descriptions + ML roles
requirements.txt                ← Pinned Python dependencies
README.md                       ← This file
```

---

## 🛠️ Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/emotionos-dashboard.git
cd emotionos-dashboard

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deployment on Streamlit Community Cloud

1. **Upload all files** to the **root of a public GitHub repository** (no sub-folders)
2. Go to **[share.streamlit.io](https://share.streamlit.io)** → New App
3. Select your repo · branch `main` · Main file: `app.py`
4. Click **Deploy** — Streamlit installs `requirements.txt` automatically

> ⚠️ `EmotionOS_Survey_Dataset.csv` must be in the **root** of the repo alongside `app.py`.

---

## 🤖 ML Models Summary

| Task | Primary | Baseline | Target Variable |
|------|---------|---------|----------------|
| Classification | Random Forest (200 trees, depth 12) | Logistic Regression | `Q24_AdoptionLikelihood` (5-class) |
| Regression | Gradient Boosting (log-transform) | Linear Regression | `Q23_BudgetINR_Numeric` |
| Clustering | K-Means++ (elbow + silhouette) | — | Unsupervised |
| Assoc. Rules | Apriori (mlxtend 0.23.1) | — | Multi-select columns |

---

## 📊 Dataset

- **2,000 synthetic respondents** — global audience (India 46%, SE Asia 15%, NA 13%, EU 10%, ME 8%)
- **52 columns** across demographics, psychographics, behaviour, feature interest, budget
- **Realistic noise** — 5% outlier injection, 4–8% missing values per selected column
- Two ML targets: 5-class categorical (adoption) + continuous numeric (budget in INR)

---

## 📄 License

MIT — free to use, modify, and distribute.

*Built for EmotionOS — The Emotional Intelligence Layer for Global Brands*
