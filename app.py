# ============================================================
#  EmotionOS — Founder's Intelligence Dashboard
#  Single-file Streamlit app  |  app.py
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os, io

from sklearn.model_selection    import train_test_split
from sklearn.ensemble           import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model       import LogisticRegression, LinearRegression
from sklearn.preprocessing      import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics            import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.cluster            import KMeans
from sklearn.decomposition      import PCA
from sklearn.impute             import SimpleImputer

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing     import TransactionEncoder
    MLXTEND_OK = True
except ImportError:
    MLXTEND_OK = False

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="EmotionOS Intelligence Dashboard",
    page_icon="🧠", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
div[data-testid="metric-container"]{
    background:linear-gradient(135deg,#1e2140,#252a45);
    border:1px solid #3a3f6b;border-radius:12px;padding:16px;}
button[data-baseweb="tab"]{font-size:13px!important;font-weight:600!important;}
</style>""", unsafe_allow_html=True)

# ============================================================
#  CONSTANTS & ORDINAL MAPS
# ============================================================
TARGET_CLASS = "Q24_AdoptionLikelihood"
TARGET_REG   = "Q23_BudgetINR_Numeric"
CLASS_NAMES  = ["Very Unlikely","Unlikely","Neutral","Likely","Very Likely"]
CLASS_COLORS = ["#ff1744","#ff6e40","#ffd740","#69f0ae","#00c853"]
ADOPTION_COLORS = dict(zip(CLASS_NAMES, CLASS_COLORS))

ORDINAL_MAPS = {
    "Q3_OrgSize":["Solo/Freelancer","2-10","11-50","51-200","201-1000","1000+"],
    "Q5_AnnualRevenue":["Under 50L/<60K","50L-2Cr/60K-240K","2Cr-10Cr/240K-1.2M","10Cr-100Cr/1.2M-12M","Above 100Cr/12M+"],
    "Q6_MarketingBudget":["Under 5L/<6K","5L-25L/6K-30K","25L-1Cr/30K-120K","1Cr-5Cr/120K-600K","Above 5Cr/600K+"],
    "Q11_EIAwareness":["Not aware at all","Slightly aware","Somewhat aware","Very aware - already use tools"],
    "Q14_CrisisFrequency":["Never - we have proactive monitoring","Rarely","Occasionally","Very frequently"],
    "Q15_CampaignBackfire":["No - campaigns always landed well","Not sure - didn't track","Yes - but minor impact","Yes - significantly damaged brand"],
    "Q18_RegionalLanguageImportance":["Not important","Somewhat important","Very important","Extremely important"],
    "Q21_CurrentSaaSSpend":["Zero - no tools","1K-10K/month","10K-50K/month","50K-2L/month","Above 2L/month"],
    "Q23_BudgetBand":["Would not pay","Under 5K","5K-15K","15K-50K","50K-2L","Above 2L"],
    "Q_TechAdoptionProfile":["Laggard - move only when forced","Late Majority - wait for industry standard","Early Majority - adopt proven solutions","Early Adopter - adopt after seeing competitor success","Innovator - adopt cutting-edge early"],
    "Q_AITrustLevel":["Don't trust AI recommendations","Skeptical but open","Use AI as one input among many","Trust AI insights and act on them"],
    "Q_StakeholderCount":["1 (just me)","2-3","4-6","7+ with formal procurement"],
    "Q_SwitchingIntent":["Have long-term contracts","Unlikely","N/A - no current tool","Likely","Very likely to switch"],
    "Q_AnalyticsReviewFrequency":["Rarely/Never","Quarterly","Monthly","Weekly","Daily"],
    "Q_AnalyticsTeamSize":["0 (no one dedicated)","1","2-5","6-15","15+"],
    "Q_CrisisResponseTime":["No process for this","Within a month","Within a week","Within 24 hours","Within hours"],
    "Q_CrisisHistory":["No - never","No - but came close","Yes - minor impact","Yes - major business impact"],
    "Q_DataPrivacyConcern":["Not concerned at all","Mildly concerned","Somewhat concerned","Very concerned - hard blocker"],
    "Q_TeamDataLiteracy":["Very low - need heavy hand-holding","Low - struggle with complex tools","Moderate - mixed team","High - most are comfortable","Very high - data-first culture"],
}

MULTI_SELECT_COLS = [
    "Q7_DataSources","Q8_AnalyticsTools","Q9_CampaignStrategies",
    "Q10_SuccessMetrics","Q13_MarketingChallenges","Q16_FeatureInterest",
    "Q17_PreferredChannels","Q19_PreferredOutputFormat",
    "Q_AdoptionBlockers","Q_CrisisImpactType","Q_RequiredIntegrations",
]
NOMINAL_COLS = [
    "Q1_Role","Q2_OrgType","Q4_Region","Q20_WhiteLabelInterest",
    "Q22_PreferredPricingModel","Q_DecisionStyle","Q_BudgetAuthority",
    "Q_AdoptionTrigger","Q_InternalBarrier","Q_CurrentSentimentTool","Q_EmotionalROIMeasured",
]
NUMERIC_COLS = [
    "Q12_EmotionalInsightRating","Q_EmotionalDataImportance","Q_BrandVsRevenuePriority",
    "Q_DataDrivenDecisionPct","Q_AnnualCampaigns","Q23_BudgetINR_Numeric",
    "Q24_AdoptionLikelihood_Ordinal","Adoption_Score_Raw",
]
CLUSTER_FEATURES = [
    "Q3_OrgSize_enc","Q5_AnnualRevenue_enc","Q6_MarketingBudget_enc",
    "Q11_EIAwareness_enc","Q12_EmotionalInsightRating","Q_TechAdoptionProfile_enc",
    "Q_AITrustLevel_enc","Q_EmotionalDataImportance","Q_BrandVsRevenuePriority",
    "Q_AnalyticsReviewFrequency_enc","Q_AnalyticsTeamSize_enc","Q_DataDrivenDecisionPct",
    "Q_CrisisHistory_enc","Q_TeamDataLiteracy_enc","Q23_BudgetINR_Numeric",
    "Q_StakeholderCount_enc","Q_DataPrivacyConcern_enc",
]
CLUSTER_PERSONAS = {
    0:("🔴 Crisis-Burnt Believers","High crisis history · Moderate budget · Fast decision","Lead with Crisis Alerts + EBS. Emphasise ROI from reputation recovery."),
    1:("🟠 Data-Forward Scalers","High data literacy · Strong AI trust · Enterprise budgets","Full platform + Segmentation Engine. Offer annual enterprise plan."),
    2:("🟡 Curious but Cautious","Moderate awareness · Mid-range budget · ROI concern","Nurture with case studies + free audit. Starter plan + upgrade path."),
    3:("🟢 Digitally Aware Late Majority","Low-moderate tech adoption · Price-sensitive","Content marketing + freemium. Focus on ease-of-use messaging."),
    4:("🔵 Agency Amplifiers","Agency type · White-label interest · High campaign volume","White-label partnership. Volume-based pricing + co-marketing."),
    5:("🟣 Silent Loyalists","High trust · Steady spend · Low churn risk","Loyalty programme + exclusive features. Upsell Segmentation Engine."),
}

# ============================================================
#  DATA LOADING & PREPROCESSING
# ============================================================
def load_data(file_obj=None):
    if file_obj is not None:
        return pd.read_csv(file_obj)
    base = os.path.dirname(os.path.abspath(__file__))
    return pd.read_csv(os.path.join(base, "EmotionOS_Survey_Dataset.csv"))

def _expand_multi(df, col):
    dummies = df[col].fillna("").str.get_dummies(sep="|")
    dummies.columns = [f"{col}__{c.replace(' ','_').replace('/','_')}" for c in dummies.columns]
    return dummies

def preprocess_data(df):
    proc = pd.DataFrame()
    encoders = {}
    for col, order in ORDINAL_MAPS.items():
        if col in df.columns:
            mapping = {v: i for i, v in enumerate(order)}
            proc[col+"_enc"] = df[col].map(mapping).fillna(-1).astype(int)
    for col in NUMERIC_COLS:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            proc[col] = s.fillna(s.median())
    for col in NOMINAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            proc[col+"_enc"] = le.fit_transform(df[col].fillna("Unknown"))
            encoders[col] = le
    multi_dfs = [_expand_multi(df, c) for c in MULTI_SELECT_COLS if c in df.columns]
    if multi_dfs:
        proc = pd.concat([proc] + multi_dfs, axis=1)
    if TARGET_CLASS in df.columns:
        tmap = {"Very Unlikely":1,"Unlikely":2,"Neutral":3,"Likely":4,"Very Likely":5}
        proc[TARGET_CLASS] = df[TARGET_CLASS].map(tmap).fillna(3).astype(int)
    if "Respondent_ID" in df.columns:
        proc["Respondent_ID"] = df["Respondent_ID"].values
    exclude = {TARGET_CLASS,TARGET_REG,"Adoption_Score_Raw",
               "Q24_AdoptionLikelihood_Ordinal","Q23_BudgetBand","Respondent_ID"}
    feature_cols = [c for c in proc.columns if c not in exclude]
    return proc, encoders, feature_cols

@st.cache_data(show_spinner="Loading dataset…")
def get_data(file_bytes):
    return load_data(io.BytesIO(file_bytes) if file_bytes else None)

@st.cache_data(show_spinner="Preprocessing…")
def get_preprocessed(_df):
    return preprocess_data(_df)

# ============================================================
#  SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=60)
    st.markdown("## 🧠 EmotionOS")
    st.markdown("**Founder's Intelligence Dashboard**")
    st.markdown("---")
    uploaded = st.file_uploader("Upload survey CSV (optional)", type=["csv"])
    st.markdown("---")
    st.markdown("""**Tabs**
- 🏠 Home · 📊 Descriptive
- 🔍 Diagnostic · 🤖 Predictive
- 🧩 Clustering · 🔗 ARM
- 🚀 New Client Predictor""")
    st.caption("© 2026 EmotionOS")

file_bytes = uploaded.read() if uploaded else None
raw_df = get_data(file_bytes)
proc_df, encoders, feature_cols = get_preprocessed(raw_df)
st.session_state["proc_df"] = proc_df
st.session_state["feature_cols"] = feature_cols

# ============================================================
#  HELPER: dark plotly layout
# ============================================================
BG = "#1e2140"
def dark(fig, h=340, t=40):
    fig.update_layout(template="plotly_dark", height=h,
                      plot_bgcolor=BG, paper_bgcolor=BG,
                      margin=dict(t=t, b=15, l=10, r=10))
    return fig

# ============================================================
#  TABS
# ============================================================
tabs = st.tabs(["🏠 Home","📊 Descriptive","🔍 Diagnostic",
                "🤖 Predictive","🧩 Clustering",
                "🔗 Association Rules","🚀 New Client Predictor"])

# ════════════════════════════════════════════════════════════
#  TAB 0 — HOME
# ════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("# 🧠 EmotionOS — Founder's Intelligence Dashboard")
    st.markdown("Data-driven GTM strategy engine · Descriptive · Diagnostic · Predictive · Prescriptive")
    st.markdown("---")

    total      = len(raw_df)
    likely_pct = round(raw_df["Q24_AdoptionLikelihood"].isin(["Likely","Very Likely"]).mean()*100,1)
    med_budget = raw_df["Q23_BudgetINR_Numeric"].dropna().median()
    top_feat   = raw_df["Q16_FeatureInterest"].dropna().str.split("|").explode().str.strip().value_counts().idxmax()
    crisis_pct = round(raw_df["Q_CrisisHistory"].isin(["Yes - major business impact","Yes - minor impact"]).mean()*100,1)
    top_seg    = raw_df["Q2_OrgType"].value_counts().idxmax()

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("👥 Respondents",    f"{total:,}")
    c2.metric("✅ Likely to Adopt", f"{likely_pct}%")
    c3.metric("💰 Median Budget",   f"₹{med_budget:,.0f}/mo")
    c4.metric("🏆 Top Feature",     top_feat.split("(")[0][:20])
    c5.metric("🔥 Had Crisis",      f"{crisis_pct}%")
    c6.metric("📦 Top Segment",     top_seg[:15])

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        order = ["Very Likely","Likely","Neutral","Unlikely","Very Unlikely"]
        cnt   = raw_df["Q24_AdoptionLikelihood"].value_counts().reindex(order).fillna(0)
        fig = go.Figure(go.Bar(x=cnt.index, y=cnt.values, marker_color=CLASS_COLORS,
                               text=cnt.values, textposition="outside"))
        st.plotly_chart(dark(fig, 300).update_layout(title="Adoption Likelihood Distribution",
                        xaxis_title="", yaxis_title="Respondents"), use_container_width=True)
    with col_b:
        rc = raw_df["Q4_Region"].value_counts().reset_index()
        rc.columns = ["Region","Count"]
        fig2 = px.pie(rc, values="Count", names="Region", hole=0.45,
                      color_discrete_sequence=px.colors.qualitative.Vivid)
        st.plotly_chart(dark(fig2, 300).update_layout(title="Respondents by Region"),
                        use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        ot = raw_df["Q2_OrgType"].value_counts().reset_index()
        ot.columns = ["OrgType","Count"]
        fig3 = px.bar(ot, x="Count", y="OrgType", orientation="h",
                      color="Count", color_continuous_scale="Viridis",
                      text="Count")
        fig3.update_traces(textposition="outside")
        fig3.update_layout(coloraxis_showscale=False,
                           yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(dark(fig3, 300).update_layout(title="Organisation Type Breakdown"),
                        use_container_width=True)
    with col_d:
        ds = raw_df["Q7_DataSources"].dropna().str.split("|").explode().str.strip().value_counts().head(7).reset_index()
        ds.columns = ["Source","Count"]
        fig4 = px.bar(ds, x="Count", y="Source", orientation="h",
                      color="Count", color_continuous_scale="Tealgrn", text="Count")
        fig4.update_traces(textposition="outside")
        fig4.update_layout(coloraxis_showscale=False,
                           yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(dark(fig4, 300).update_layout(title="Top Data Sources Used"),
                        use_container_width=True)

# ════════════════════════════════════════════════════════════
#  TAB 1 — DESCRIPTIVE
# ════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("# 📊 Descriptive Analysis")
    st.markdown("---")

    # EI Maturity by Org Type
    ei_org = raw_df.groupby("Q2_OrgType")["Q12_EmotionalInsightRating"].mean().dropna().sort_values().reset_index()
    ei_org.columns = ["OrgType","Avg EI"]
    fig = px.bar(ei_org, x="Avg EI", y="OrgType", orientation="h",
                 color="Avg EI", color_continuous_scale="RdYlGn",
                 text=ei_org["Avg EI"].round(2), title="Avg EI Capability by Org Type (1=Low, 5=High)")
    fig.update_traces(textposition="outside")
    fig.update_layout(coloraxis_showscale=False, yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(dark(fig, 320), use_container_width=True)

    # Feature demand heatmap
    st.markdown("### 🔥 Feature Demand Heatmap by Org Type")
    feats = ["Emotional Brand Score (EBS)","Omnichannel Sentiment Dashboard",
             "Predictive Mood & Crisis Alerts","Campaign Emotion Optimizer","Emotional Customer Segmentation"]
    flabels = ["EBS","Sentiment Dash","Crisis Alerts","Campaign Opt","Segmentation"]
    orgs = raw_df["Q2_OrgType"].value_counts().index.tolist()
    mat = []
    for org in orgs:
        sub = raw_df[raw_df["Q2_OrgType"]==org]
        mat.append([round(sub["Q16_FeatureInterest"].dropna().str.contains(f.split("(")[0].strip(),regex=False).mean()*100,1) for f in feats])
    fig2 = px.imshow(mat, x=flabels, y=orgs, color_continuous_scale="Viridis",
                     aspect="auto", text_auto=".1f", title="Feature Interest % by Org Type")
    st.plotly_chart(dark(fig2, 380), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        b_org = raw_df.groupby("Q2_OrgType")["Q23_BudgetINR_Numeric"].median().dropna().sort_values(ascending=False).reset_index()
        b_org.columns = ["OrgType","Median"]
        fig3 = px.bar(b_org, x="Median", y="OrgType", orientation="h",
                      color="Median", color_continuous_scale="Teal",
                      text=b_org["Median"].apply(lambda x: f"₹{x:,.0f}"),
                      title="Median Budget by Org Type")
        fig3.update_traces(textposition="outside")
        fig3.update_layout(coloraxis_showscale=False, yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(dark(fig3, 320), use_container_width=True)
    with col_b:
        ch = raw_df["Q_CrisisHistory"].value_counts().reset_index()
        ch.columns = ["Crisis","Count"]
        fig4 = px.pie(ch, values="Count", names="Crisis", hole=0.4,
                      color_discrete_sequence=["#ff1744","#ff6e40","#ffd740","#00e676"],
                      title="Brand Crisis History")
        st.plotly_chart(dark(fig4, 320), use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        pricing = raw_df["Q22_PreferredPricingModel"].value_counts().reset_index()
        pricing.columns = ["Model","Count"]
        fig5 = px.bar(pricing, x="Count", y="Model", orientation="h",
                      color="Count", color_continuous_scale="Purples",
                      text="Count", title="Preferred Pricing Model")
        fig5.update_traces(textposition="outside")
        fig5.update_layout(coloraxis_showscale=False, yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(dark(fig5, 300), use_container_width=True)
    with col_d:
        chal = raw_df["Q13_MarketingChallenges"].dropna().str.split("|").explode().str.strip().value_counts().head(7).reset_index()
        chal.columns = ["Challenge","Count"]
        chal["Short"] = chal["Challenge"].str[:32]+"…"
        fig6 = px.bar(chal, x="Count", y="Short", orientation="h",
                      color="Count", color_continuous_scale="Reds",
                      text="Count", title="Top Marketing Challenges")
        fig6.update_traces(textposition="outside")
        fig6.update_layout(coloraxis_showscale=False, yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(dark(fig6, 300), use_container_width=True)

    tools = raw_df["Q8_AnalyticsTools"].dropna().str.split("|").explode().str.strip().value_counts().reset_index()
    tools.columns = ["Tool","Count"]
    fig7 = px.treemap(tools, path=["Tool"], values="Count", color="Count",
                      color_continuous_scale="Blues", title="Analytics Tools in Use")
    st.plotly_chart(dark(fig7, 360), use_container_width=True)

    with st.expander("📋 View Raw Data (first 100 rows)"):
        st.dataframe(raw_df.head(100), use_container_width=True)

# ════════════════════════════════════════════════════════════
#  TAB 2 — DIAGNOSTIC
# ════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("# 🔍 Diagnostic Analysis")
    st.markdown("---")

    # Correlation with adoption score
    num_cols = ["Q12_EmotionalInsightRating","Q_EmotionalDataImportance",
                "Q_BrandVsRevenuePriority","Q_DataDrivenDecisionPct",
                "Q_AnnualCampaigns","Q_CurrentToolSatisfaction","Adoption_Score_Raw"]
    avail = [c for c in num_cols if c in raw_df.columns]
    corr_df = raw_df[avail].apply(pd.to_numeric, errors="coerce").dropna()
    if "Adoption_Score_Raw" in corr_df:
        corrs = corr_df.corr()["Adoption_Score_Raw"].drop("Adoption_Score_Raw").sort_values()
        labels = {"Q12_EmotionalInsightRating":"EI Self-Rating","Q_EmotionalDataImportance":"Emotional Data Importance",
                  "Q_BrandVsRevenuePriority":"Brand vs Revenue","Q_DataDrivenDecisionPct":"Data-Driven %",
                  "Q_AnnualCampaigns":"Annual Campaigns","Q_CurrentToolSatisfaction":"Current Tool Satisfaction"}
        cdf = pd.DataFrame({"Feature":[labels.get(i,i) for i in corrs.index],"Corr":corrs.values,
                             "Color":["#ff6e40" if v<0 else "#69f0ae" for v in corrs.values]})
        fig = px.bar(cdf, x="Corr", y="Feature", orientation="h",
                     color="Color", color_discrete_map="identity",
                     title="Pearson Correlation with Adoption Score")
        fig.add_vline(x=0, line_dash="dash", line_color="white")
        fig.update_layout(showlegend=False)
        st.plotly_chart(dark(fig, 320), use_container_width=True)

    # Barrier drill-down
    st.markdown("### 🚧 Why Brands Won't Adopt")
    unlikely = raw_df[raw_df["Q24_AdoptionLikelihood"].isin(["Unlikely","Very Unlikely"])]
    col_a, col_b = st.columns(2)
    with col_a:
        bar = unlikely["Q_InternalBarrier"].value_counts(normalize=True).mul(100).round(1).reset_index()
        bar.columns = ["Barrier","Pct"]
        fig2 = px.bar(bar, x="Pct", y="Barrier", orientation="h",
                      color="Pct", color_continuous_scale="Reds",
                      text=bar["Pct"].apply(lambda x: f"{x:.1f}%"),
                      title=f"Top Barriers (N={len(unlikely)} non-adopters)")
        fig2.update_traces(textposition="outside")
        fig2.update_layout(coloraxis_showscale=False, yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(dark(fig2, 320), use_container_width=True)
    with col_b:
        blk = unlikely["Q_AdoptionBlockers"].dropna().str.split("|").explode().str.strip().value_counts(normalize=True).mul(100).round(1).head(7).reset_index()
        blk.columns = ["Blocker","Pct"]
        fig3 = px.pie(blk, values="Pct", names="Blocker", hole=0.35,
                      color_discrete_sequence=px.colors.qualitative.Bold,
                      title="Specific Adoption Blockers")
        st.plotly_chart(dark(fig3, 320), use_container_width=True)

    # Segment comparator
    st.markdown("### 🔄 Segment Comparator")
    seg_col = st.selectbox("Compare by:", ["Q2_OrgType","Q4_Region","Q3_OrgSize","Q_TechAdoptionProfile"],
        format_func=lambda x:{"Q2_OrgType":"Org Type","Q4_Region":"Region","Q3_OrgSize":"Org Size","Q_TechAdoptionProfile":"Tech Profile"}.get(x,x))
    opts = sorted(raw_df[seg_col].dropna().unique())
    cs1, cs2 = st.columns(2)
    s1 = cs1.selectbox("Segment A", opts, index=0)
    s2 = cs2.selectbox("Segment B", opts, index=min(1,len(opts)-1))
    mets = {"Avg EI Rating":"Q12_EmotionalInsightRating","Data-Driven %":"Q_DataDrivenDecisionPct",
            "Avg Budget":"Q23_BudgetINR_Numeric","Annual Campaigns":"Q_AnnualCampaigns"}
    d1, d2 = raw_df[raw_df[seg_col]==s1], raw_df[raw_df[seg_col]==s2]
    rows = [{"Metric":l,"A":pd.to_numeric(d1[c],errors="coerce").median(),"B":pd.to_numeric(d2[c],errors="coerce").median()} for l,c in mets.items() if c in raw_df.columns]
    cmp = pd.DataFrame(rows)
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(name=s1[:20], x=cmp["Metric"], y=cmp["A"], marker_color="#7c4dff"))
    fig4.add_trace(go.Bar(name=s2[:20], x=cmp["Metric"], y=cmp["B"], marker_color="#00b0ff"))
    fig4.update_layout(barmode="group", title=f"{s1[:20]} vs {s2[:20]}")
    st.plotly_chart(dark(fig4, 320), use_container_width=True)

    # Crisis → Adoption pipeline
    st.markdown("### 🔥 Crisis History → Adoption Pipeline")
    crisis_order = ["No - never","No - but came close","Yes - minor impact","Yes - major business impact"]
    ca = raw_df.groupby(["Q_CrisisHistory","Q24_AdoptionLikelihood"]).size().reset_index(name="Count")
    ca["Q_CrisisHistory"] = pd.Categorical(ca["Q_CrisisHistory"], categories=crisis_order, ordered=True)
    fig5 = px.bar(ca.sort_values("Q_CrisisHistory"), x="Q_CrisisHistory", y="Count",
                  color="Q24_AdoptionLikelihood", barmode="stack",
                  color_discrete_map=ADOPTION_COLORS,
                  category_orders={"Q24_AdoptionLikelihood":CLASS_NAMES},
                  title="Adoption Likelihood by Crisis History")
    st.plotly_chart(dark(fig5, 360), use_container_width=True)

# ════════════════════════════════════════════════════════════
#  TAB 3 — PREDICTIVE
# ════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("# 🤖 Predictive Models")
    st.markdown("---")

    def get_Xy_cls(proc, feats):
        X = proc[feats].copy(); y = proc[TARGET_CLASS].copy()
        m = y.notna(); X, y = X[m], y[m].astype(int)-1
        imp = SimpleImputer(strategy="median")
        X = pd.DataFrame(imp.fit_transform(X), columns=feats)
        return X, y, imp

    def get_Xy_reg(proc, feats):
        X = proc[feats].copy(); y = proc[TARGET_REG].copy()
        m = y.notna()&(y>=0); X, y = X[m], y[m]
        imp = SimpleImputer(strategy="median")
        X = pd.DataFrame(imp.fit_transform(X), columns=feats)
        return X, np.log1p(y), y, imp

    @st.cache_resource(show_spinner="Training models…")
    def train_all(h):
        pf  = st.session_state["proc_df"]
        fc  = st.session_state["feature_cols"]
        X,y,imp_c = get_Xy_cls(pf, fc)
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
        rf = RandomForestClassifier(n_estimators=200,max_depth=12,random_state=42,n_jobs=-1)
        rf.fit(Xtr,ytr)
        sc = StandardScaler(); Xtr_s=sc.fit_transform(Xtr); Xte_s=sc.transform(Xte)
        lr = LogisticRegression(max_iter=1000,random_state=42,C=0.5); lr.fit(Xtr_s,ytr)
        Xl,yl,yr,imp_r = get_Xy_reg(pf, fc)
        Xltr,Xlte,yltr,ylte = train_test_split(Xl,yl,test_size=0.2,random_state=42)
        _,_,yrtr,yrte = train_test_split(Xl,yr,test_size=0.2,random_state=42)
        gb = GradientBoostingRegressor(n_estimators=200,max_depth=5,learning_rate=0.05,random_state=42)
        gb.fit(Xltr,yltr)
        return rf,lr,sc,imp_c,Xte,yte,gb,imp_r,Xlte,ylte,yrte

    dh = hash(str(proc_df.shape))
    rf,lr,sc_cls,imp_cls,Xte,yte,gb,imp_reg,Xlte,ylte,yrte = train_all(dh)

    sub_cls, sub_reg = st.tabs(["🎯 Classification","💰 Regression"])

    with sub_cls:
        st.markdown("### Model Performance")
        rf_p=rf.predict(Xte); lr_p=lr.predict(sc_cls.transform(Xte))
        rf_pr=rf.predict_proba(Xte); lr_pr=lr.predict_proba(sc_cls.transform(Xte))

        def mrow(name,yt,yp):
            return {"Model":name,"Accuracy":f"{accuracy_score(yt,yp):.3f}",
                    "Precision":f"{precision_score(yt,yp,average='weighted',zero_division=0):.3f}",
                    "Recall":f"{recall_score(yt,yp,average='weighted',zero_division=0):.3f}",
                    "F1 (weighted)":f"{f1_score(yt,yp,average='weighted',zero_division=0):.3f}"}
        st.dataframe(pd.DataFrame([mrow("Random Forest",yte,rf_p),mrow("Logistic Regression",yte,lr_p)]),
                     use_container_width=True, hide_index=True)

        st.markdown("### Confusion Matrices")
        cm1,cm2 = st.columns(2)
        for cm_data, title, col_obj in [(confusion_matrix(yte,rf_p,labels=list(range(5))),"Random Forest",cm1),
                                         (confusion_matrix(yte,lr_p,labels=list(range(5))),"Logistic Regression",cm2)]:
            fig,ax = plt.subplots(figsize=(5,4))
            fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
            sns.heatmap(cm_data,annot=True,fmt="d",cmap="YlOrRd",
                        xticklabels=CLASS_NAMES,yticklabels=CLASS_NAMES,ax=ax)
            ax.set_title(title,color="white"); ax.set_xlabel("Predicted",color="white")
            ax.set_ylabel("Actual",color="white"); ax.tick_params(colors="white",rotation=30)
            plt.tight_layout(); col_obj.pyplot(fig); plt.close(fig)

        st.markdown("### Per-Class Report (Random Forest)")
        rep = classification_report(yte,rf_p,target_names=CLASS_NAMES,output_dict=True,zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.iloc[:-3].round(3), use_container_width=True)

        st.markdown("### ROC Curves — Random Forest (One-vs-Rest)")
        yte_bin = label_binarize(yte, classes=list(range(5)))
        fig_roc = go.Figure()
        for i,(cn,cc) in enumerate(zip(CLASS_NAMES,CLASS_COLORS)):
            if i < rf_pr.shape[1]:
                fpr,tpr,_ = roc_curve(yte_bin[:,i], rf_pr[:,i])
                fig_roc.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",
                    name=f"{cn} (AUC={auc(fpr,tpr):.2f})",line=dict(color=cc,width=2)))
        fig_roc.add_shape(type="line",x0=0,y0=0,x1=1,y1=1,line=dict(dash="dash",color="gray"))
        fig_roc.update_layout(xaxis_title="FPR",yaxis_title="TPR",title="ROC Curves (OvR)")
        st.plotly_chart(dark(fig_roc,400), use_container_width=True)

        st.markdown("### Top 20 Feature Importances")
        imp_ser = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False).head(20)
        fig_fi = px.bar(x=imp_ser.values, y=[c.replace("_enc","").replace("Q_","") for c in imp_ser.index],
                        orientation="h", color=imp_ser.values, color_continuous_scale="Viridis",
                        text=imp_ser.values.round(4))
        fig_fi.update_traces(textposition="outside")
        fig_fi.update_layout(coloraxis_showscale=False, yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(dark(fig_fi, 500), use_container_width=True)

        st.markdown("### 🔥 Top 50 Warm Leads")
        X_all = pd.DataFrame(imp_cls.transform(proc_df[feature_cols]), columns=feature_cols)
        adopt_prob = rf.predict_proba(X_all)[:,3] + rf.predict_proba(X_all)[:,4]
        leads = raw_df.copy(); leads["Adoption_Probability"] = adopt_prob
        leads = leads.sort_values("Adoption_Probability", ascending=False).head(50)
        dcols = [c for c in ["Respondent_ID","Q2_OrgType","Q4_Region","Q23_BudgetBand",
                              "Q24_AdoptionLikelihood","Adoption_Probability"] if c in leads.columns]
        leads[dcols[-1]] = leads[dcols[-1]].round(3)
        st.dataframe(leads[dcols], use_container_width=True, hide_index=True)

    with sub_reg:
        st.markdown("### Model Performance")
        gb_p = np.expm1(gb.predict(Xlte)); yr_vals = np.expm1(ylte.values)
        lr_reg = LinearRegression()
        sc_reg = StandardScaler()
        lr_reg.fit(sc_reg.fit_transform(Xlte), ylte)
        lr_p_r = np.expm1(lr_reg.predict(sc_reg.transform(Xlte)))

        def rrrow(name,yt,yp):
            return {"Model":name,"RMSE":f"₹{np.sqrt(mean_squared_error(yt,yp)):,.0f}",
                    "MAE":f"₹{mean_absolute_error(yt,yp):,.0f}","R²":f"{r2_score(yt,yp):.4f}"}
        st.dataframe(pd.DataFrame([rrrow("Gradient Boosting",yr_vals,gb_p),
                                    rrrow("Linear Regression",yr_vals,lr_p_r)]),
                     use_container_width=True, hide_index=True)

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            pva = pd.DataFrame({"Actual":np.clip(yr_vals,0,400000),"Predicted":np.clip(gb_p,0,400000)})
            fig_pv = px.scatter(pva, x="Actual", y="Predicted", opacity=0.5,
                                color_discrete_sequence=["#7c4dff"], title="Predicted vs Actual")
            mv = pva.max().max()
            fig_pv.add_shape(type="line",x0=0,y0=0,x1=mv,y1=mv,line=dict(color="#ffd740",dash="dash"))
            st.plotly_chart(dark(fig_pv, 360), use_container_width=True)
        with col_r2:
            res = yr_vals - gb_p
            fig_res = px.scatter(x=gb_p, y=res, opacity=0.45,
                                 color_discrete_sequence=["#ff6e40"],
                                 labels={"x":"Predicted","y":"Residual"},
                                 title="Residuals vs Predicted")
            fig_res.add_hline(y=0, line_dash="dash", line_color="white")
            st.plotly_chart(dark(fig_res, 360), use_container_width=True)

        gb_imp = pd.Series(gb.feature_importances_, index=feature_cols).sort_values(ascending=False).head(20)
        fig_gr = px.bar(x=gb_imp.values, y=[c.replace("_enc","") for c in gb_imp.index],
                        orientation="h", color=gb_imp.values, color_continuous_scale="Sunset",
                        text=gb_imp.values.round(4), title="Feature Importances — Regression")
        fig_gr.update_traces(textposition="outside")
        fig_gr.update_layout(coloraxis_showscale=False, yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(dark(fig_gr, 500), use_container_width=True)

# ════════════════════════════════════════════════════════════
#  TAB 4 — CLUSTERING
# ════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("# 🧩 Customer Clustering")
    st.markdown("---")

    @st.cache_data(show_spinner="Computing elbow…")
    def elbow_data(dh):
        pf = st.session_state["proc_df"]
        af = [c for c in CLUSTER_FEATURES if c in pf.columns]
        Xc = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(pf[af]), columns=af)
        Xs = StandardScaler().fit_transform(Xc)
        kr = range(2,10)
        ins=[]; sils=[]
        for k in kr:
            km=KMeans(n_clusters=k,n_init=10,random_state=42); lb=km.fit_predict(Xs)
            ins.append(km.inertia_)
            from sklearn.metrics import silhouette_score
            sils.append(silhouette_score(Xs,lb,sample_size=500))
        return list(kr),ins,sils,Xs,af

    dh = hash(str(proc_df.shape))
    kr, ins, sils, Xs, af = elbow_data(dh)

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        fe = go.Figure(go.Scatter(x=list(kr),y=ins,mode="lines+markers",
                                   line=dict(color="#7c4dff",width=2),marker=dict(size=8)))
        fe.update_layout(title="Elbow Curve",xaxis_title="K",yaxis_title="Inertia")
        st.plotly_chart(dark(fe,280), use_container_width=True)
    with col_e2:
        fs = go.Figure(go.Scatter(x=list(kr),y=sils,mode="lines+markers",
                                   line=dict(color="#00c853",width=2),marker=dict(size=8)))
        fs.update_layout(title="Silhouette Score (higher=better)",xaxis_title="K",yaxis_title="Score")
        st.plotly_chart(dark(fs,280), use_container_width=True)

    best_k = kr[int(np.argmax(sils))]
    k = st.slider("Select K:", 2, 8, min(best_k,5))

    @st.cache_resource(show_spinner="Clustering…")
    def run_km(dh, k):
        pf = st.session_state["proc_df"]
        af2 = [c for c in CLUSTER_FEATURES if c in pf.columns]
        imp2 = SimpleImputer(strategy="median")
        sc2  = StandardScaler()
        Xc2  = pd.DataFrame(imp2.fit_transform(pf[af2]), columns=af2)
        Xs2  = sc2.fit_transform(Xc2)
        km2  = KMeans(n_clusters=k, n_init=20, random_state=42)
        labs = km2.fit_predict(Xs2)
        X2d  = PCA(n_components=2, random_state=42).fit_transform(Xs2)
        return labs, X2d, km2, sc2, imp2, af2

    labs, X2d, km2, sc2, imp2, af2 = run_km(dh, k)
    st.session_state.update({"cluster_model":km2,"cluster_scaler":sc2,
                              "cluster_imp":imp2,"cluster_feats":af2})

    raw_aug = raw_df.copy(); raw_aug["Cluster"] = labs
    pca_df = pd.DataFrame({"PC1":X2d[:,0],"PC2":X2d[:,1],"Cluster":labs.astype(str),
                            "OrgType":raw_df["Q2_OrgType"].values,"Budget":raw_df["Q23_BudgetINR_Numeric"].values})
    fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster",
                         hover_data=["OrgType","Budget"],
                         color_discrete_sequence=px.colors.qualitative.Bold,
                         title=f"Customer Clusters — PCA 2D (K={k})", opacity=0.65)
    fig_pca.update_traces(marker=dict(size=5))
    st.plotly_chart(dark(fig_pca,460), use_container_width=True)

    st.markdown("### 🃏 Cluster Persona Cards")
    card_cols = st.columns(min(k, 3))
    for i, cid in enumerate(sorted(raw_aug["Cluster"].unique())):
        sub = raw_aug[raw_aug["Cluster"]==cid]
        nm, desc, tip = CLUSTER_PERSONAS.get(cid,(f"Cluster {cid}","Mixed profile","Analyse further"))
        with card_cols[i % 3]:
            st.markdown(f"""
<div style="background:linear-gradient(135deg,#1e2140,#252a45);border:1px solid #3a3f6b;
border-radius:14px;padding:14px;margin-bottom:10px;">
<h4 style="color:#e8eaf6;margin:0 0 4px">{nm}</h4>
<p style="color:#9e9ec0;font-size:11px;margin:0 0 6px">{desc}</p>
<b style="color:#ffd740">N={len(sub):,}</b> · <b style="color:#69f0ae">₹{sub["Q23_BudgetINR_Numeric"].median():,.0f}/mo</b><br/>
<small style="color:#9e9ec0">Top: {sub["Q2_OrgType"].value_counts().idxmax()}<br/>
Adopt rate: {sub["Q24_AdoptionLikelihood"].isin(["Likely","Very Likely"]).mean()*100:.1f}%</small>
<hr style="border-color:#3a3f6b;margin:6px 0"/>
<small style="color:#80cbc4">💡 {tip}</small></div>""", unsafe_allow_html=True)

    csv_cl = raw_aug.to_csv(index=False).encode()
    st.download_button("⬇️ Download Clustered Dataset", csv_cl,
                       "EmotionOS_Clustered.csv", "text/csv")

# ════════════════════════════════════════════════════════════
#  TAB 5 — ASSOCIATION RULE MINING
# ════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("# 🔗 Association Rule Mining")
    st.markdown("---")

    if not MLXTEND_OK:
        st.error("⚠️ `mlxtend` not installed. Add `mlxtend==0.23.1` to requirements.txt and redeploy.")
    else:
        with st.expander("⚙️ Thresholds"):
            ct1,ct2,ct3 = st.columns(3)
            mn_s = ct1.slider("Min Support",    0.05,0.50,0.15,0.01)
            mn_c = ct2.slider("Min Confidence", 0.30,0.90,0.55,0.05)
            mn_l = ct3.slider("Min Lift",        1.0, 5.0, 1.2,0.1)

        st.info(f"Support ≥ {mn_s:.2f} · Confidence ≥ {mn_c:.2f} · Lift ≥ {mn_l:.1f}")

        def run_arm(trans, ms, mc, ml):
            te = TransactionEncoder(); arr = te.fit_transform(trans)
            df_enc = pd.DataFrame(arr, columns=te.columns_)
            freq = apriori(df_enc, min_support=ms, use_colnames=True, low_memory=True)
            if freq.empty: return pd.DataFrame()
            rules = association_rules(freq, metric="lift", min_threshold=ml)
            rules = rules[rules["confidence"]>=mc].copy()
            rules["antecedents"] = rules["antecedents"].apply(lambda x:", ".join(sorted(x)))
            rules["consequents"] = rules["consequents"].apply(lambda x:", ".join(sorted(x)))
            return rules.sort_values("lift",ascending=False).reset_index(drop=True)

        def show_rules(rules, title):
            if rules.empty:
                st.warning("No rules found. Try lowering thresholds.")
                return
            top = rules.head(20).copy()
            top["Rule"] = top["antecedents"].str[:25]+" → "+top["consequents"].str[:25]
            fig = px.scatter(top, x="support", y="confidence", size="lift", color="lift",
                             hover_data=["antecedents","consequents","lift"],
                             color_continuous_scale="Viridis", size_max=30, title=title)
            st.plotly_chart(dark(fig,400), use_container_width=True)
            disp = rules.head(20)[["antecedents","consequents","support","confidence","lift"]].round(4)
            disp.columns = ["Antecedents","Consequents","Support","Confidence","Lift"]
            st.dataframe(disp, use_container_width=True, hide_index=True)
            st.markdown("**Plain-English Top 10:**")
            for _, r in rules.head(10).iterrows():
                st.markdown(f"- Brands wanting **{r['antecedents'][:50]}** also want **{r['consequents'][:50]}** — Conf: {r['confidence']:.0%}, Lift: {r['lift']:.2f}×")

        arm_a, arm_b, arm_c = st.tabs(["🔷 Feature Bundles","⚡ Pain → Feature","📡 Source → Channel"])

        with arm_a:
            st.markdown("## 🔷 Feature Bundle Mining")
            t_a = raw_df["Q16_FeatureInterest"].dropna().str.split("|").apply(lambda x:[i.strip() for i in x]).tolist()
            with st.spinner("Running Apriori…"):
                r_a = run_arm(t_a, mn_s, mn_c, mn_l)
            show_rules(r_a, "Feature Co-occurrence — Support vs Confidence")

        with arm_b:
            st.markdown("## ⚡ Pain Point → Feature Mapping")
            t_b=[]
            for _, row in raw_df[["Q13_MarketingChallenges","Q16_FeatureInterest"]].iterrows():
                items = (["PAIN:"+i.strip() for i in row["Q13_MarketingChallenges"].split("|") if pd.notna(row["Q13_MarketingChallenges"]) and i.strip()] +
                         ["FEAT:"+i.strip() for i in row["Q16_FeatureInterest"].split("|") if pd.notna(row["Q16_FeatureInterest"]) and i.strip()])
                if items: t_b.append(items)
            with st.spinner("Mining…"):
                r_b = run_arm(t_b, mn_s*0.8, mn_c*0.85, mn_l)
            if not r_b.empty:
                r_b2 = r_b[r_b["antecedents"].str.startswith("PAIN:")&r_b["consequents"].str.startswith("FEAT:")].copy()
                r_b2["antecedents"] = r_b2["antecedents"].str.replace("PAIN:","",regex=False)
                r_b2["consequents"] = r_b2["consequents"].str.replace("FEAT:","",regex=False)
                show_rules(r_b2, "Pain → Feature Demand")
            else:
                st.warning("No rules found. Try lowering thresholds.")

        with arm_c:
            st.markdown("## 📡 Data Source → Channel Preference")
            t_c=[]
            for _, row in raw_df[["Q7_DataSources","Q17_PreferredChannels"]].iterrows():
                items = (["SRC:"+i.strip() for i in row["Q7_DataSources"].split("|") if pd.notna(row["Q7_DataSources"]) and i.strip()] +
                         ["CHAN:"+i.strip() for i in row["Q17_PreferredChannels"].split("|") if pd.notna(row["Q17_PreferredChannels"]) and i.strip()])
                if items: t_c.append(items)
            with st.spinner("Mining…"):
                r_c = run_arm(t_c, mn_s*0.8, mn_c*0.85, mn_l)
            if not r_c.empty:
                r_c2 = r_c[r_c["antecedents"].str.startswith("SRC:")&r_c["consequents"].str.startswith("CHAN:")].copy()
                r_c2["antecedents"] = r_c2["antecedents"].str.replace("SRC:","",regex=False)
                r_c2["consequents"] = r_c2["consequents"].str.replace("CHAN:","",regex=False)
                show_rules(r_c2, "Data Source → Preferred Channel")
            else:
                st.warning("No rules found. Try lowering thresholds.")

# ════════════════════════════════════════════════════════════
#  TAB 6 — NEW CLIENT PREDICTOR
# ════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("# 🚀 New Client Predictor")
    st.markdown("Upload a CSV **or** fill the form → get adoption prediction, budget forecast, cluster & strategy card.")
    st.markdown("---")

    @st.cache_resource(show_spinner="Preparing predictor models…")
    def pred_models(dh):
        pf = st.session_state["proc_df"]; fc = st.session_state["feature_cols"]
        X,y,imp_c = get_Xy_cls(pf,fc)
        rfp = RandomForestClassifier(n_estimators=200,max_depth=12,random_state=42,n_jobs=-1)
        rfp.fit(X,y)
        Xl,yl,_,imp_r = get_Xy_reg(pf,fc)
        gbp = GradientBoostingRegressor(n_estimators=200,max_depth=5,learning_rate=0.05,random_state=42)
        gbp.fit(Xl,yl)
        return rfp, gbp, imp_c, imp_r

    rfp, gbp, imp_cp, imp_rp = pred_models(hash(str(proc_df.shape)))

    def predict_row(row_proc, fc):
        X = row_proc[fc].copy()
        Xc = pd.DataFrame(imp_cp.transform(X), columns=fc)
        Xr = pd.DataFrame(imp_rp.transform(X), columns=fc)
        proba = rfp.predict_proba(Xc)[0]
        label = CLASS_NAMES[int(np.argmax(proba))]
        budget = float(np.expm1(gbp.predict(Xr)[0]))
        return label, proba, budget

    def strategy_card(org, region, label, budget, proba, cid=None):
        urg = max(0, min(10, round(proba[4]*20+proba[3]*10+proba[2]*3+proba[1]*(-3)+proba[0]*(-8),1)))
        tier = "Enterprise" if budget>=100000 else "Growth" if budget>=30000 else "Starter"
        price= "₹2,00,000+/mo" if tier=="Enterprise" else "₹50,000/mo" if tier=="Growth" else "₹15,000/mo"
        feats= ("EBS + Dashboard + Crisis Alerts + Segmentation" if tier=="Enterprise" else
                "EBS + Dashboard + Campaign Optimizer" if tier=="Growth" else "EBS + Basic Dashboard")
        pname,_,ptip = CLUSTER_PERSONAS.get(cid,(f"Cluster {cid}","","Analyse further")) if cid is not None else ("—","","—")
        uc = "#00c853" if urg>=7 else "#ffd740" if urg>=4 else "#ff6e40"
        st.markdown(f"""
<div style="background:linear-gradient(135deg,#1a2035,#1e2a45);border:1px solid #3a3f6b;
border-radius:16px;padding:22px;margin:10px 0">
<h3 style="color:#e8eaf6;margin:0 0 4px">🚀 Sales Strategy Card</h3>
<p style="color:#9e9ec0;margin:0 0 14px">{org} · {region}</p>
<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:14px">
<div style="background:#252a45;border-radius:10px;padding:10px 18px;flex:1;min-width:110px">
<div style="color:#9e9ec0;font-size:11px">ADOPTION</div>
<div style="color:#ffd740;font-size:18px;font-weight:700">{label}</div></div>
<div style="background:#252a45;border-radius:10px;padding:10px 18px;flex:1;min-width:110px">
<div style="color:#9e9ec0;font-size:11px">BUDGET</div>
<div style="color:#69f0ae;font-size:18px;font-weight:700">₹{budget:,.0f}/mo</div></div>
<div style="background:#252a45;border-radius:10px;padding:10px 18px;flex:1;min-width:110px">
<div style="color:#9e9ec0;font-size:11px">TIER</div>
<div style="color:#80cbc4;font-size:18px;font-weight:700">{tier}</div>
<div style="color:#9e9ec0;font-size:11px">{price}</div></div>
<div style="background:#252a45;border-radius:10px;padding:10px 18px;flex:1;min-width:110px">
<div style="color:#9e9ec0;font-size:11px">URGENCY</div>
<div style="color:{uc};font-size:18px;font-weight:700">{urg}/10</div></div></div>
<b style="color:#e8eaf6">🎯 Pitch:</b> <span style="color:#cfd8dc">{feats}</span><br/><br/>
<b style="color:#e8eaf6">🧩 Tribe:</b> <span style="color:#cfd8dc">{pname}</span><br/>
<span style="color:#80cbc4"><i>💡 {ptip}</i></span></div>""", unsafe_allow_html=True)

        fig_p = go.Figure(go.Bar(x=CLASS_NAMES,y=[round(p*100,1) for p in proba],
                                  marker_color=CLASS_COLORS,
                                  text=[f"{p*100:.1f}%" for p in proba],textposition="outside"))
        fig_p.update_layout(title="Adoption Probability Distribution",yaxis_title="Probability (%)")
        st.plotly_chart(dark(fig_p,280), use_container_width=True)

    up_tab, form_tab = st.tabs(["📂 CSV Upload (Bulk)","📝 Manual Entry (Single)"])

    with up_tab:
        st.markdown("Upload a CSV matching the EmotionOS survey schema.")
        tmpl = raw_df.head(0).to_csv(index=False).encode()
        st.download_button("⬇️ Download Template CSV", tmpl, "template.csv","text/csv")
        up_file = st.file_uploader("Upload new clients CSV", type=["csv"], key="up_new")
        if up_file:
            new_df = pd.read_csv(up_file)
            st.success(f"✅ {len(new_df):,} records loaded")
            st.dataframe(new_df.head(5), use_container_width=True)
            with st.spinner("Predicting…"):
                np2, _, fc2 = preprocess_data(new_df)
                for c in feature_cols:
                    if c not in np2.columns: np2[c]=0
                np2 = np2[feature_cols]
                Xnc = pd.DataFrame(imp_cp.transform(np2), columns=feature_cols)
                Xnr = pd.DataFrame(imp_rp.transform(np2), columns=feature_cols)
                all_p = rfp.predict_proba(Xnc)
                labels_new = [CLASS_NAMES[i] for i in np.argmax(all_p,axis=1)]
                budgets_new = np.expm1(gbp.predict(Xnr))
                res = new_df.copy()
                res["Predicted_Adoption"] = labels_new
                res["Adoption_Confidence"] = np.max(all_p,axis=1).round(3)
                res["P_Likely+"] = (all_p[:,3]+all_p[:,4]).round(3)
                res["Predicted_Budget_INR"] = budgets_new.round(0).astype(int)
                km_m = st.session_state.get("cluster_model")
                if km_m:
                    kf = st.session_state.get("cluster_feats",[])
                    kf_av = [c for c in kf if c in np2.columns]
                    X_km = st.session_state["cluster_imp"].transform(np2[kf_av].fillna(0))
                    res["Cluster"] = km_m.predict(st.session_state["cluster_scaler"].transform(X_km))
                    res["Persona"] = res["Cluster"].map(lambda x: CLUSTER_PERSONAS.get(x,("?","",""))[0])
            dcols = [c for c in ["Respondent_ID","Q2_OrgType","Q4_Region","Predicted_Adoption",
                                  "Adoption_Confidence","P_Likely+","Predicted_Budget_INR","Persona"] if c in res.columns]
            st.dataframe(res[dcols], use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download Predictions", res.to_csv(index=False).encode(),
                               "predictions.csv","text/csv")

    with form_tab:
        st.markdown("Fill in what you know — unknowns are auto-imputed.")
        with st.form("pred_form"):
            fa,fb,fc_ = st.columns(3)
            org_t = fa.selectbox("Org Type",["D2C Brand","FMCG/CPG","E-commerce Platform","Marketing/Creative Agency","SaaS/Tech Startup","Enterprise/MNC","SME","Non-Profit/Government"])
            org_s = fb.selectbox("Org Size",["Solo/Freelancer","2-10","11-50","51-200","201-1000","1000+"],index=2)
            reg   = fc_.selectbox("Region",["India-Metro","India-Tier2/3","Southeast Asia","Middle East","Europe","North America","Other"])
            fd,fe = st.columns(2)
            rev   = fd.selectbox("Revenue",["Under 50L/<60K","50L-2Cr/60K-240K","2Cr-10Cr/240K-1.2M","10Cr-100Cr/1.2M-12M","Above 100Cr/12M+"],index=1)
            mktb  = fe.selectbox("Marketing Budget",["Under 5L/<6K","5L-25L/6K-30K","25L-1Cr/30K-120K","1Cr-5Cr/120K-600K","Above 5Cr/600K+"],index=1)
            ff,fg,fh = st.columns(3)
            tap   = ff.selectbox("Tech Adoption",["Innovator - adopt cutting-edge early","Early Adopter - adopt after seeing competitor success","Early Majority - adopt proven solutions","Late Majority - wait for industry standard","Laggard - move only when forced"],index=1)
            ait   = fg.selectbox("AI Trust",["Trust AI insights and act on them","Use AI as one input among many","Skeptical but open","Don't trust AI recommendations"],index=1)
            eia   = fh.selectbox("EI Awareness",["Very aware - already use tools","Somewhat aware","Slightly aware","Not aware at all"],index=1)
            fi,fj,fk = st.columns(3)
            crh   = fi.selectbox("Crisis History",["Yes - major business impact","Yes - minor impact","No - but came close","No - never"])
            dtl   = fj.selectbox("Data Literacy",["Very high - data-first culture","High - most are comfortable","Moderate - mixed team","Low - struggle with complex tools","Very low - need heavy hand-holding"],index=2)
            ddp   = fk.slider("Data-Driven %",0,100,50,5)
            fl,fm = st.columns(2)
            eir   = fl.slider("EI Rating (1-5)",1,5,2)
            anc   = fm.number_input("Annual Campaigns",1,200,15)
            go_btn = st.form_submit_button("🔮 Predict & Generate Strategy Card")

        if go_btn:
            defaults = {
                "Q2_OrgType":org_t,"Q3_OrgSize":org_s,"Q4_Region":reg,
                "Q5_AnnualRevenue":rev,"Q6_MarketingBudget":mktb,
                "Q_TechAdoptionProfile":tap,"Q_AITrustLevel":ait,
                "Q11_EIAwareness":eia,"Q_CrisisHistory":crh,
                "Q_TeamDataLiteracy":dtl,"Q_DataDrivenDecisionPct":float(ddp),
                "Q12_EmotionalInsightRating":float(eir),"Q_AnnualCampaigns":float(anc),
                "Q1_Role":"Marketing Manager","Q18_RegionalLanguageImportance":"Somewhat important",
                "Q20_WhiteLabelInterest":"N/A - not an agency","Q21_CurrentSaaSSpend":"1K-10K/month",
                "Q22_PreferredPricingModel":"Monthly subscription","Q23_BudgetBand":"15K-50K",
                "Q_DecisionStyle":"Data/case studies","Q_BudgetAuthority":"CMO/VP",
                "Q_StakeholderCount":"2-3","Q_AdoptionTrigger":"A specific data gap",
                "Q_InternalBarrier":"Price/ROI justification",
                "Q_CurrentSentimentTool":"No - didn't know such tools existed",
                "Q_CurrentToolSatisfaction":0,"Q_SwitchingIntent":"Likely",
                "Q_AnalyticsReviewFrequency":"Monthly","Q_AnalyticsTeamSize":"1",
                "Q_CrisisResponseTime":"Within a week","Q_EmotionalROIMeasured":"No - but would like to",
                "Q_DataPrivacyConcern":"Somewhat concerned","Q_EmotionalDataImportance":3.0,
                "Q_BrandVsRevenuePriority":3.0,"Q14_CrisisFrequency":"Occasionally",
                "Q15_CampaignBackfire":"Not sure - didn't track","Q23_BudgetINR_Numeric":0.0,
                "Q24_AdoptionLikelihood":"Neutral","Q24_AdoptionLikelihood_Ordinal":3,
                "Adoption_Score_Raw":10.0,
                "Q7_DataSources":"Social Media|Website/App Analytics",
                "Q8_AnalyticsTools":"Google Analytics/Looker Studio",
                "Q9_CampaignStrategies":"Performance/Paid Advertising|Content Marketing",
                "Q10_SuccessMetrics":"ROI/Revenue Generated|Engagement Rate",
                "Q13_MarketingChallenges":"Understanding WHY customers feel the way they do",
                "Q16_FeatureInterest":"Emotional Brand Score (EBS)|Omnichannel Sentiment Dashboard",
                "Q17_PreferredChannels":"Instagram/Facebook|Google/App Store Reviews",
                "Q19_PreferredOutputFormat":"Real-time dashboard with visual charts",
                "Q_AdoptionBlockers":"Cost|ROI uncertainty","Q_CrisisImpactType":"",
                "Q_RequiredIntegrations":"Google Data Studio/Looker|Meta Ads Manager",
            }
            nr = pd.DataFrame([defaults])
            np3, _, fc3 = preprocess_data(nr)
            for c in feature_cols:
                if c not in np3.columns: np3[c]=0
            np3 = np3[feature_cols]
            label_s, proba_s, budget_s = predict_row(np3, feature_cols)
            cid_s = None
            km_m = st.session_state.get("cluster_model")
            if km_m:
                kf = st.session_state.get("cluster_feats",[])
                kf_av = [c for c in kf if c in np3.columns]
                X_km = st.session_state["cluster_imp"].transform(np3[kf_av].fillna(0))
                cid_s = int(km_m.predict(st.session_state["cluster_scaler"].transform(X_km))[0])
            strategy_card(org_t, reg, label_s, budget_s, proba_s, cid_s)
