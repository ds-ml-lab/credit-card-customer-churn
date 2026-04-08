# Credit Card Customer Churn

## Overview

This project implements an end-to-end data science workflow to address customer churn in a credit card business context, combining statistical analysis and machine learning to model attrition while uncovering the behavioral drivers embedded in the [BankChurners](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers) dataset. Beyond predictive performance, the solution integrates a Retrieval-Augmented Generation (RAG) system to automate prescriptive action plans, ensuring that retention efforts are not only accurate but fully aligned with corporate policies and business rules.

---

## Repository Structure

```text
credit-card-customer-churn/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling_evaluation.ipynb
│
├── model/
│   └── churn_model.pickle
│
├── data/
│   ├── BankChurners.csv
│   ├── BankChurners_clean.csv
│   └── BankChurners_processed.csv
│
├── customer_retention_policy.pdf
├── app.py
├── .env.example
├── requirements.txt
└── README.md
```

---

## Workflow

- `01_eda.ipynb`: Analysis of data structure, feature distributions and potential data quality issues, with a focus on their relationship to customer churn.
- `02_preprocessing.ipynb`: Operationalization of EDA insights through feature engineering and consistent handling of missing values and data inconsistencies.
- `03_modeling_evaluation.ipynb`: Baseline model comparison to identify the most suitable architecture, followed by stratified cross-validation, hyperparameter optimization and final evaluation on a held-out test set.

---

## Model Performance

The final model achieved strong performance on the churn class when evaluated on unseen data:

- **Precision:** 0.93
- **Recall:** 0.90
- **F1 Score:** 0.92

A Recall of 90% ensures that the vast majority of potential churners are captured by the model, while a Precision of 93% ensures that retention resources are not wasted on satisfied customers.

---

## Deployment

The final model is deployed via a **Streamlit** application designed for real-time risk assessment and prescriptive action planning. 

- **Predictive Engine:** The interface focuses on the key behavioral drivers identified during the interpretability (SHAP) analysis, running an XGBoost algorithm in the background to calculate the exact Churn Propensity Score (CPS).
- **Prescriptive Engine (RAG):** Based on the calculated risk, a Retrieval-Augmented Generation system powered by **Llama 3.3 70B** (via Groq API) is triggered. It cross-references the customer's profile with the vectorized internal policy document (`customer_retention_policy.pdf`) stored in **ChromaDB**, instantly outputting a customized, budget-approved retention offer and communication strategy.

[**Live Dashboard**](https://credit-card-customer-churn.streamlit.app)

### Local Usage Instructions

1. Clone the repository:
```bash
git clone [https://github.com/ds-ml-lab/credit-card-customer-churn.git](https://github.com/ds-ml-lab/credit-card-customer-churn.git)
cd credit-card-customer-churn
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
Rename the `.env.example` file to `.env` and insert your Groq API key:
```text
GROQ_API_KEY=your_api_key_here
```

4. Run the application:
```bash
streamlit run app.py
```

---

## Important Notes

1. This repository is intended for educational purposes only. The dataset is used under fair use guidelines for non-commercial study and skill demonstration.

2. This project was developed with the assistance of LLMs to enhance productivity in specific areas:
    - Structuring and brainstorming ideas.
    - Refining English text for clarity and tone.
    - Debugging and syntax correction.
    - Structuring the policy document and the Streamlit interface.
    
---

## Author

**Pedro Siqueira** [LinkedIn](https://www.linkedin.com/in/phenriquels/) | [GitHub](https://github.com/phenriquels01)