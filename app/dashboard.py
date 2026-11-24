import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

# Configure dashboard
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

# API address
API_URL = "http://127.0.0.1:8000"

# Required transaction features
FEATURE_COLUMNS = [
    f"V{i}" for i in range(1, 29)
] + ["Amount"]

# Dashboard title
st.title("Intelligent Financial Fraud Detection")

st.write(
    "Machine learning-based credit card fraud detection "
    "using Tuned XGBoost."
)

st.divider()

# Sidebar
st.sidebar.title("Model Information")

st.sidebar.write("Model: Tuned XGBoost")
st.sidebar.write("Threshold: 0.50")
st.sidebar.write("Input Features: 29")

st.sidebar.divider()

st.sidebar.subheader("API Connection")

try:

    response = requests.get(
        API_URL + "/health",
        timeout=5
    )

    response.raise_for_status()

    st.sidebar.success("API Connected")

except requests.exceptions.RequestException:

    st.sidebar.error("API Not Connected")

st.sidebar.write(
    "Backend: FastAPI"
)

st.sidebar.write(
    "Frontend: Streamlit"
)

# CSV upload section
st.header("Transaction Upload")

st.write(
    "Upload a CSV file containing transaction features."
)

# Create empty CSV template
template = pd.DataFrame(
    columns=FEATURE_COLUMNS
)

st.download_button(
    label="Download CSV Template",
    data=template.to_csv(index=False),
    file_name="fraud_detection_template.csv",
    mime="text/csv"
)

uploaded_file = st.file_uploader(
    "Choose transaction CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    uploaded_data = pd.read_csv(uploaded_file)

    st.write("Uploaded Dataset Preview")

    st.dataframe(
        uploaded_data.head(),
        use_container_width=True
    )

    st.write(
        "Total Records:", len(uploaded_data)
    )

if uploaded_file is not None:

    # Check missing columns
    missing_columns = [
        column for column in FEATURE_COLUMNS
        if column not in uploaded_data.columns
    ]

    # Allow Time and Class as optional metadata
    allowed_columns = (
        FEATURE_COLUMNS + ["Time", "Class"]
    )

    # Check unexpected columns
    extra_columns = [
        column for column in uploaded_data.columns
        if column not in allowed_columns
    ]

    if missing_columns:

        st.error(
            "Missing required columns: "
            + str(missing_columns)
        )

    elif extra_columns:

        st.error(
            "Unexpected columns found: "
            + str(extra_columns)
        )

    elif len(uploaded_data) == 0:

        st.error("The uploaded CSV is empty.")

    elif len(uploaded_data) > 5000:

        st.error(
            "Maximum upload size is 5000 transactions."
        )

    else:

        # Select required model features
        transaction_data = uploaded_data[
            FEATURE_COLUMNS
        ].copy()

        # Convert features into numerical values
        for column in FEATURE_COLUMNS:

            transaction_data[column] = pd.to_numeric(
                transaction_data[column],
                errors="coerce"
            )

        # Check missing values
        if transaction_data.isnull().any().any():

            st.error(
                "The dataset contains missing or invalid values."
            )

        elif not np.isfinite(
            transaction_data.to_numpy()
        ).all():

            st.error(
                "The dataset contains infinite values."
            )

        else:

            st.success(
                "Dataset validation successful!"
            )

            st.session_state["transaction_data"] = (
                transaction_data
            )

            st.session_state["uploaded_data"] = (
                uploaded_data
            )

# Prediction section
if "transaction_data" in st.session_state:

    transaction_data = st.session_state[
        "transaction_data"
    ]

    st.divider()

    st.header("Fraud Detection")

    if st.button(
        "Run Fraud Detection",
        type="primary"
    ):

        with st.spinner("Analysing transactions..."):

            try:

                # Convert transactions into records
                transactions = (
                    transaction_data.to_dict(
                        orient="records"
                    )
                )

                # Send request to FastAPI
                response = requests.post(
                    API_URL + "/predict/batch",
                    json={
                        "transactions": transactions
                    },
                    timeout=120
                )

                response.raise_for_status()

                result = response.json()

                # Get prediction results
                predictions = pd.DataFrame(
                    result["predictions"]
                )

                # Combine original data and predictions
                results = st.session_state[
                    "uploaded_data"
                ][
                    ["Time", "Class"]
                ].copy() if all(
                    column in st.session_state["uploaded_data"].columns
                    for column in ["Time", "Class"]
                ) else pd.DataFrame(
                    index=range(len(predictions))
                )

                if (
                    "Time" in st.session_state["uploaded_data"].columns
                    and "Time" not in results.columns
                ):
                    results["Time"] = (
                        st.session_state["uploaded_data"]["Time"]
                    )

                results.insert(
                    0,
                    "Transaction_ID",
                    range(1, len(predictions) + 1)
                )

                results["Fraud_Probability"] = (
                    predictions["fraud_probability"]
                )

                results["Predicted_Class"] = (
                    predictions["predicted_class"]
                )

                results["Predicted_Label"] = (
                    predictions["predicted_label"]
                )

                if "Class" in results.columns:

                    results = results.rename(
                        columns={"Class": "Actual_Class"}
                    )

                # Save results
                st.session_state["results"] = results

                st.success(
                    "Transaction analysis completed!"
                )

            except requests.exceptions.RequestException as error:

                st.error(
                    "Unable to communicate with the API."
                )

                st.error(str(error))

# Display results
if "results" in st.session_state:

    results = st.session_state["results"]

    st.divider()

    st.header("Prediction Summary")

    # Count predictions
    total_transactions = len(results)

    fraud_count = (
        results["Predicted_Class"] == 1
    ).sum()

    legitimate_count = (
        results["Predicted_Class"] == 0
    ).sum()

    fraud_percentage = (
        fraud_count / total_transactions
    ) * 100

    # Create four metric cards
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        total_transactions
    )

    col2.metric(
        "Fraud Detected",
        fraud_count
    )

    col3.metric(
        "Legitimate",
        legitimate_count
    )

    col4.metric(
        "Fraud Rate",
        f"{fraud_percentage:.2f}%"
    )

    st.divider()

    # Prediction distribution
    st.subheader("Transaction Classification")

    chart_data = pd.DataFrame({
        "Transaction Type": [
            "Legitimate",
            "Fraudulent"
        ],
        "Count": [
            legitimate_count,
            fraud_count
        ]
    })

    st.bar_chart(
        chart_data,
        x="Transaction Type",
        y="Count"
    )

    # Display detailed predictions
    st.subheader("Transaction Results")

    st.dataframe(
        results.sort_values(
            by="Fraud_Probability",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    # Download results
    csv_data = results.to_csv(index=False)

    st.download_button(
        label="Download Prediction Results",
        data=csv_data,
        file_name="fraud_predictions.csv",
        mime="text/csv"
    )

# Individual transaction explanation
if "results" in st.session_state:

    st.divider()

    st.header("Transaction Explainability")

    results = st.session_state["results"]

    transaction_data = st.session_state[
        "transaction_data"
    ]

    # Allow user to select transaction
    selected_transaction = st.selectbox(
        "Select Transaction to Explain",
        options=list(range(len(results))),
        format_func=lambda index: (
            "Transaction "
            + str(index + 1)
            + " - "
            + results.iloc[index]["Predicted_Label"]
            + " ("
            + f"{results.iloc[index]['Fraud_Probability']:.2%}"
            + ")"
        )
    )

    if st.button("Generate SHAP Explanation"):

        with st.spinner("Generating explanation..."):

            try:

                # Select transaction features
                features = transaction_data.iloc[
                    selected_transaction
                ].to_dict()

                # Request SHAP explanation
                response = requests.post(
                    API_URL + "/explain",
                    json={"features": features},
                    timeout=60
                )

                response.raise_for_status()

                explanation = response.json()

                # Display prediction
                st.subheader("Prediction Explanation")

                col1, col2 = st.columns(2)

                col1.metric(
                    "Fraud Probability",
                    f"{explanation['fraud_probability']:.2%}"
                )

                col2.metric(
                    "Predicted Result",
                    explanation["predicted_label"]
                )

                # Get top SHAP features
                shap_data = pd.DataFrame(
                    explanation["top_features"]
                )

                shap_data = shap_data.sort_values(
                    by="shap_value"
                )

                st.subheader("Top Feature Contributions")

                # Create SHAP chart
                fig, ax = plt.subplots(figsize=(9, 5))

                colors = []

                for value in shap_data["shap_value"]:

                    if value > 0:
                        colors.append("#D9534F")
                    else:
                        colors.append("#27845B")

                ax.barh(
                    shap_data["feature"],
                    shap_data["shap_value"],
                    color=colors
                )

                ax.axvline(
                    x=0,
                    color="black",
                    linewidth=0.8
                )

                ax.set_xlabel(
                    "SHAP Value (Raw Log-Odds)"
                )

                ax.set_title(
                    "Features Influencing This Prediction"
                )

                plt.tight_layout()

                st.pyplot(fig)

                plt.close(fig)

                st.dataframe(
                    pd.DataFrame(
                        explanation["top_features"]
                    ),
                    use_container_width=True,
                    hide_index=True
                )

                st.info(
                    "Positive SHAP values push the model "
                    "towards fraud. Negative values push "
                    "the prediction towards legitimate."
                )

            except requests.exceptions.RequestException as error:

                st.error(
                    "Unable to generate SHAP explanation."
                )

                st.error(str(error))
