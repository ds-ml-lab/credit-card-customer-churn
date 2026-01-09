import streamlit as st
import pickle
import pandas as pd

# 1. Page Configuration & Custom CSS
st.set_page_config(
    page_title="Credit Card Retention Engine",
    layout="centered"
)

st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .stButton>button {
        width: 100%;
        font-weight: bold;
    }
    h1 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown(
    """
    <h1 style="text-align:center;">Credit Card Retention Engine</h1>
    <p style="text-align:center; color: #555;">
    Predictive analytics based on behavioral drivers.<br>
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# 2. Load Model Artifact
@st.cache_resource
def load_artifact():
    with open("model/churn_model.pickle", "rb") as f:
        return pickle.load(f)

try:
    artifact = load_artifact()
    model = artifact["model"]
    model_columns = artifact["columns"]
    model_defaults = artifact["defaults"]
except Exception as e:
    st.error(f"System Error: Model artifact not found. {e}")
    st.stop()

# 3. Sidebar — Controls
st.sidebar.markdown("### Input Parameters")

total_trans_ct = st.sidebar.slider(
    "Total Transactions Count",
    min_value=10, max_value=140, value=60,
    help="Frequency of usage in the last year."
)

total_trans_amt = st.sidebar.number_input(
    "Total Transaction Amount",
    min_value=500.0, max_value=18500.0, value=4000.0, step=100.0,
    help="Total monetary value spent in the last year."
)

total_revolving_bal = st.sidebar.number_input(
    "Total Revolving Balance",
    min_value=0, max_value=2500, value=1000, step=100,
    help="Unpaid balance carried over."
)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("RUN PREDICTION", type="primary")

# 4. Prediction Engine
if predict_btn:
    # --- Feature Engineering ---
    calculated_avg_ticket = (
        total_trans_amt / total_trans_ct if total_trans_ct > 0 else 0.0
    )

    user_input = {
        "Total_Trans_Ct": total_trans_ct,
        "Total_Trans_Amt": total_trans_amt,
        "Avg_Ticket_Size": calculated_avg_ticket,
        "Total_Revolving_Bal": total_revolving_bal
    }

    X_user = pd.DataFrame([user_input])
    X_user = X_user.reindex(columns=model_columns)
    X_user = X_user.fillna(model_defaults)

    probability = model.predict_proba(X_user)[0][1]

    # 5. Results Dashboard
    if probability < 0.30:
        status_color = "#28a745"
        status_label = "LOW RISK"
        recommendation = "Maintain current relationship strategy."
    elif probability < 0.70:
        status_color = "#ffc107"
        status_label = "MODERATE RISK"
        recommendation = "Monitor closely. Consider engagement incentives."
    else:
        status_color = "#dc3545"
        status_label = "HIGH RISK"
        recommendation = "Urgent retention action required."

    st.markdown("### Risk Assessment")

    with st.container(border=True):
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(
                f"""
                <h2 style='color: {status_color}; margin:0; padding:0;'>{status_label}</h2>
                """,
                unsafe_allow_html=True
            )
            st.markdown(f"**Action:** {recommendation}")

        with col2:
            st.metric(label="Churn Probability", value=f"{probability:.1%}")

    st.caption(f"Average Ticket Size: ${calculated_avg_ticket:,.2f}")

else:
    st.info("Configure customer parameters on the sidebar and click 'Run Prediction'.")