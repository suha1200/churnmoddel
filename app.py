import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st
import matplotlib
import plotly.figure_factory as ff
import plotly.express as px

# Set the backend of matplotlib to the 'agg' backend to avoid any issues related to GUI backends in streamlit
matplotlib.use('Agg')

# Define utility functions for the Streamlit app
def get_value(val, my_dict):
    """Retrieve value from a dictionary given a key."""
    # If the key is found in the dictionary, return its value
    return my_dict.get(val, None)

def get_key(val, my_dict):
    """Retrieve key from a dictionary given a value."""
    # Iterate through the dictionary to find the key for the given value
    for key, value in my_dict.items():
        if val == value:
            return key
    # If the value is not found, return None
    return None

def load_model_n_predict(model_file):
    """Load a pre-trained model for prediction."""
    # Load a model from the specified file path and return it
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model

# Define the main function to construct the Streamlit app
def main():
    # Set the title of the app
    st.title('Customer Churn Prediction Tool')

    # Create a sidebar for navigation with two always visible options
    # Users can choose between "Exploratory Data Analysis" and "Prediction"
    st.sidebar.title("Navigation")
    activity = ["Exploratory Data Analysis", "Prediction"]
    choice = st.sidebar.radio("Choose an Activity", activity)

    # Load and preprocess the data
    # Data is read from a CSV file, irrelevant columns are dropped, and categorical columns are factorized
    df = pd.read_csv('data/Churn_Modelling.csv')
    data_cleaned = df.drop(['CustomerId', 'Surname'], axis=1)
    data_cleaned.dropna(inplace=True)  # Remove any rows with missing values
    data_cleaned['Geography'] = pd.factorize(data_cleaned['Geography'])[0] + 1  # Encode 'Geography'
    data_cleaned['Gender'] = pd.factorize(data_cleaned['Gender'])[0] + 1  # Encode 'Gender'

    if choice == "Exploratory Data Analysis":
        # Display the header for the EDA section
        st.header("Exploratory Data Analysis (EDA)")
        st.markdown(
            "Explore the dataset to understand the distribution of various features and their relation to customer churn."
        )

        # Use an expander for previewing the dataset
        with st.expander("Preview Dataset"):
            # Allow the user to specify the number of rows to display
            number = st.number_input("Number of Rows to Show", min_value=5, max_value=100, value=10)
            # Display the top specified number of rows from the dataframe
            st.dataframe(df.head(number))

        # Use an expander for descriptive statistics
        with st.expander("Show Descriptive Statistics"):
            # Show descriptive statistics of the dataframe
            st.write(df.describe())

        # Use an expander to show the shape of the dataset
        with st.expander("Show Dataset Shape"):
            # Display the number of rows and columns in the dataframe
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

        # Use an expander for value counts of a selected column
        with st.expander("Show Value Counts for a Column"):
            # Let the user select a column to display value counts
            column = st.selectbox("Column to Display", df.columns)
            # Display the value counts for the selected column
            st.write(df[column].value_counts())

        # Use an expander for the correlation matrix heatmap
        with st.expander("Show Correlation Matrix Heatmap"):
            # Calculate the correlation matrix
            corr_matrix = df.corr()
            # Create a heatmap using the correlation matrix
            fig = ff.create_annotated_heatmap(
                z=corr_matrix.values,
                x=list(corr_matrix.columns),
                y=list(corr_matrix.index),
                annotation_text=corr_matrix.round(2).values,
                colorscale='RdBu',
                reversescale=True,
                showscale=True
            )
            # Update the layout of the figure for better presentation
            fig.update_layout(margin=dict(t=50, l=200))
            fig.update_layout(height=600, width=800)
            # Display the heatmap
            st.plotly_chart(fig, use_container_width=True)

        # Use an expander to show the distribution of the 'Age' column
        with st.expander("Show Age Distribution"):
            # Create a histogram of the 'Age' column with a marginal boxplot
            fig = px.histogram(df, x='Age', nbins=20, marginal="box", title="Distribution of Customer Ages")
            # Update the layout of the histogram
            fig.update_layout(bargap=0.1)
            # Display the histogram
            st.plotly_chart(fig, use_container_width=True)

    if choice == 'Prediction':
        st.subheader("Prediction Section")
        st.markdown("""
            Predict the likelihood of a customer leaving the bank using their profile information.
            Fill out the customer details below and press "Predict" to see the outcome.
            """)

        # Mapping dictionaries for geography and gender
        d_geography = {'France': 0, 'Spain': 1, 'Germany': 2, 'Other': 3}
        d_gender = {'Female': 0, 'Male': 1}

        # Using columns to logically group inputs into two columns
        col1, col2 = st.columns(2)

        with col1:
            credit_score = st.slider("Credit Score", 350, 850,
                                     help="Customer's credit score (350-850). A higher score indicates better creditworthiness.")
            age = st.slider("Age", 18, 100, help="Customer's age.")
            balance = st.number_input("Balance", min_value=0.0, max_value=999999.0, format="%.2f",
                                      help="Customer's account balance.")
            geography = st.selectbox("Location", tuple(d_geography.keys()), help="Customer's country of residence.")
            has_cr_card = st.checkbox("Has Credit Card", help="Does the customer have a credit card?")

        with col2:
            tenure = st.slider("Tenure", 0, 10, help="Number of years the customer has been with the bank.")
            no_products = st.slider("Number of Products", 0, 10, help="Number of bank products the customer uses.")
            gender = st.radio("Gender", tuple(d_gender.keys()), help="Customer's gender.")
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
            prediction_proba = model_predictor.predict_proba(sample_data_df)[0]
            fig = px.bar(x=['No Churn', 'Churn'], y=prediction_proba, labels={'x': 'Outcome', 'y': 'Probability'},
                         title='Churn Prediction Probability')
            st.plotly_chart(fig, use_container_width=True)

            prediction = model_predictor.predict(sample_data_df)
            pred_result = "Churn" if prediction[0] == 1 else "No Churn"
            st.success(f"Prediction Result: {pred_result}")
            st.markdown("""
                            **What does this mean?** A prediction of **"Churn"** suggests the customer is likely to leave the bank. Consider strategies to increase customer satisfaction and retention.
                            """)


if __name__ == '__main__':
    main()
