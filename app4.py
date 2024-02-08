import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for matplotlib to avoid GUI issues within Streamlit


# Utility Functions
def get_value(val, my_dict):
    """Retrieve value from a dictionary given a key."""
    return my_dict.get(val, None)


def get_key(val, my_dict):
    """Retrieve key from a dictionary given a value."""
    for key, value in my_dict.items():
        if val == value:
            return key
    return None


def load_model_n_predict(model_file):
    """Load a pre-trained model for prediction."""
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model


def main():
    st.title('Customer Churn Prediction Tool')

    # Sidebar for navigation
    activity = ["Exploratory Data Analysis", "Prediction"]
    choice = st.sidebar.selectbox("Choose an Activity", activity)

    # Load and preprocess data
    df = pd.read_csv('data/Churn_Modelling.csv')
    data_cleaned = df.drop(['CustomerId', 'Surname'], axis=1)
    data_cleaned.dropna(inplace=True)
    data_cleaned['Geography'] = pd.factorize(data_cleaned['Geography'])[0] + 1
    data_cleaned['Gender'] = pd.factorize(data_cleaned['Gender'])[0] + 1

    if choice == "Exploratory Data Analysis":
        st.header("Exploratory Data Analysis (EDA)")
        st.markdown(
            "Explore the dataset to understand the distribution of various features and their relation to customer churn.")

        # Data Preview
        if st.checkbox("Preview Dataset", help="Check to preview the dataset."):
            number = st.number_input("Number of Rows to Show", min_value=5, max_value=100, value=10,
                                     help="Choose the number of rows to display.")
            st.dataframe(df.head(number))

        # Descriptive Statistics
        if st.checkbox("Show Descriptive Statistics", help="Check to display the dataset's descriptive statistics."):
            st.write(df.describe())

        # Dataset Shape
        if st.checkbox("Show Dataset Shape", help="Check to display the shape of the dataset."):
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

        # Column-wise Value Counts
        if st.checkbox("Show Value Counts for a Column", help="Select a column to display its value counts."):
            column = st.selectbox("Column", df.columns, help="Select a column to see value counts.")
            st.write(df[column].value_counts())

        # Correlation Matrix Heatmap
        if st.checkbox("Show Correlation Matrix Heatmap", help="Check to display a heatmap of the correlation matrix."):
            plt.figure(figsize=(10, 7))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
            st.pyplot()

        # Additional EDA: Distribution of Customer Age
        if st.checkbox("Show Age Distribution", help="Check to display the distribution of customer ages."):
            plt.figure(figsize=(8, 6))
            sns.histplot(df['Age'], bins=20, kde=True)
            plt.title("Distribution of Customer Ages")
            st.pyplot()

    if choice == 'Prediction':
        st.subheader("Prediction Section")
        st.markdown("""
            Predict the likelihood of a customer leaving the bank using their profile information.
            Fill out the customer details below and press "Predict" to see the outcome.
            """)

        # Mapping dictionaries for geography and gender
        d_geography = {'France': 0, 'Spain': 1, 'Germany': 2, 'Other': 3}
        d_gender = {'Female': 0, 'Male': 1}

        # Using columns to logically group inputs
        col1, col2, col3 = st.columns(3)

        with col1:
            credit_score = st.slider("Credit Score", 350, 850,
                                     help="Customer's credit score (350-850). A higher score indicates better creditworthiness.")
            age = st.slider("Age", 18, 100, help="Customer's age.")
            balance = st.number_input("Balance", min_value=0.0, max_value=999999.0, format="%.2f",
                                      help="Customer's account balance.")

        with col2:
            geography = st.selectbox("Location", tuple(d_geography.keys()), help="Customer's country of residence.")
            tenure = st.slider("Tenure", 0, 10, help="Number of years the customer has been with the bank.")
            no_products = st.slider("Number of Products", 0, 10, help="Number of bank products the customer uses.")

        with col3:
            gender = st.radio("Gender", tuple(d_gender.keys()), help="Customer's gender.")
            has_cr_card = st.checkbox("Has Credit Card", help="Does the customer have a credit card?")
            is_active_member = st.checkbox("Is Active Member", help="Is the customer an active member?")

        salary = st.number_input("Estimated Salary", min_value=0.0, help="Customer's estimated salary.")

        # Encoding the inputs
        k_gender = get_value(gender, d_gender)
        k_geography = get_value(geography, d_geography)

        # Preparing the data for prediction
        vectorized_result = [credit_score, k_geography, k_gender, age, tenure, balance, no_products, int(has_cr_card),
                             int(is_active_member), salary]
        sample_data_df = pd.DataFrame([vectorized_result],
                                      columns=['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
                                               'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])

        if st.button("Predict"):
            model_predictor = load_model_n_predict("churn_modelling_pipeline_v3.pkl")
            prediction = model_predictor.predict(sample_data_df)
            pred_result = "Churn" if prediction[0] == 1 else "No Churn"
            st.success(f"Prediction Result: {pred_result}")
            st.markdown("""
                **What does this mean?** A prediction of **"Churn"** suggests the customer is likely to leave the bank. Consider strategies to increase customer satisfaction and retention.
                """)


if __name__ == '__main__':
    main()
