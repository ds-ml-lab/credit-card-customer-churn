import streamlit as st
import pickle
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

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
    Predictive analytics based on behavioral drivers & Prescriptive AI.<br>
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

# 3. RAG
@st.cache_resource(show_spinner=False)
def setup_rag():
    pdf_path = "customer_retention_policy.pdf"
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File {pdf_path} not found in the directory.")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    split_texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(split_texts, embeddings)

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    if not GROQ_API_KEY:
        st.error("Error: GROQ_API_KEY not found in the .env file.")
        st.stop()
    
    llm = ChatGroq(
        temperature=0.3, 
        groq_api_key=GROQ_API_KEY, 
        model_name="llama-3.3-70b-versatile"
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    prompt = ChatPromptTemplate.from_template(
        """Answer the question using ONLY the provided context.

Context:
{context}

Question:
{input}
"""
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, document_chain)

    return qa_chain

# 4. Sidebar — Controls
st.sidebar.markdown("### Input Parameters")

total_trans_ct = st.sidebar.slider(
    "Total Transactions Count",
    min_value=10, max_value=140, value=60,
)

total_trans_amt = st.sidebar.number_input(
    "Total Transaction Amount",
    min_value=500.0, max_value=18500.0, value=4000.0, step=100.0,
)

calculated_avg_ticket = (total_trans_amt / total_trans_ct) if total_trans_ct > 0 else 0.0

st.sidebar.text_input(
    "Average Ticket Size",
    value=f"{calculated_avg_ticket:,.2f}",
    disabled=True
)

total_revolving_bal = st.sidebar.number_input(
    "Total Revolving Balance",
    min_value=0, max_value=2500, value=1000, step=100,
)

st.sidebar.divider()
predict_btn = st.sidebar.button("RUN PREDICTION", type="primary")

# 5. Prediction Engine
if predict_btn:
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

    # 6. Results Dashboard
    if probability < 0.30:
        status_color = "#28a745"
        status_label = "LOW RISK"
        recommendation = "This customer shows strong engagement and low churn risk. No immediate retention action is required."
    elif probability < 0.70:
        status_color = "#ffc107"
        status_label = "MODERATE RISK"
        recommendation = "Monitor closely. Consider engagement incentives below."
    else:
        status_color = "#dc3545"
        status_label = "HIGH RISK"
        recommendation = "Urgent retention action required below."

    st.markdown("### Risk Assessment")

    with st.container(border=True):
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(
                f"<h2 style='color: {status_color}; margin:0;'>{status_label}</h2>",
                unsafe_allow_html=True
            )
            st.markdown(f"{recommendation}")

        with col2:
            st.metric(label="Churn Probability", value=f"{probability:.1%}")

    if probability >= 0.30: 
        st.divider()
        st.markdown("### AI Retention Strategy")
        
        with st.spinner("Analyzing internal policy and generating custom action plan..."):
            try:
                qa_chain = setup_rag()
                
                prompt_text = f"""
The predictive model analyzed the data and officially classified this customer as: {status_label} (Churn Propensity Score: {probability:.1%}).

Current customer profile:
- Transaction count: {total_trans_ct}
- Total amount spent: USD {total_trans_amt:,.2f}
- Revolving balance: USD {total_revolving_bal:,.2f}
- Average ticket: USD {calculated_avg_ticket:,.2f}

You are an expert retention strategist. Based ONLY on the rules from the internal retention policy document and the data above, provide a direct action plan. 

Generate the response EXACTLY in this format:

<b>DIAGNOSIS:</b> (Write the evaluation of the metrics. State the primary behavioral driver. Do not invent backstories).

<b>RISK CLASSIFICATION & MIC:</b> (Write EXACTLY "{status_label}" and the Maximum Intervention Cost limit associated with it from the policy).

<b>AUTHORIZED OFFER:</b> (Write the exact program name and applied rule. Ensure the offer respects the MIC limit).

<b>CHANNEL STRATEGY:</b> (Write the channel name and applied rule, followed by a period.).
<ul>
  <li>(Tactical tip 1)</li>
  <li>(Tactical tip 2)</li>
</ul>

CRITICAL FORMATTING RULES:
1. LANGUAGE: English only.
2. OBJECTIVITY: ALWAYS be objective and direct. No introductory or concluding filler text.
3. INLINE TEXT: The text for Diagnosis, Risk, Offer, and Channel Strategy MUST have a line break after the colon.
4. NO SYMBOLS: Never use '$', use 'USD'.
"""
                
                response = qa_chain.invoke({"input": prompt_text})

                clean_text = response["answer"]
                
                st.markdown(
                    f"""
                    <p align="justify">
                        {clean_text}
                    </p>
                    """, 
                    unsafe_allow_html=True
                )
                
            except FileNotFoundError as fnf_error:
                st.error(str(fnf_error))
            except Exception as e:
                st.error(f"Local AI Error: {e}")

else:
    st.info("Configure customer parameters on the sidebar and click 'Run Prediction'.")