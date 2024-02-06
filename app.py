import os
from pycaret.classification import *
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import catboost

matplotlib.use('Agg')


def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value


# Find the Key From Dictionary
def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key


# Load Models
def load_model_n_predict(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model


def main():
    st.title('Churn Modelling')
    activity = ["eda", "prediction"]
    choice = st.sidebar.selectbox("choose  An Activity", activity)
    df = pd.read_csv('data/Churn_Modelling.csv')
    # Drop 'CustomerId' and 'Surname' columns
    data_cleaned = df.drop(['CustomerId', 'Surname'], axis=1)

    # Drop all rows with missing values
    data_cleaned.dropna(inplace=True)

    # Convert 'Geography' and 'Gender' to integers
    # We use factorize to convert unique strings to an enumerated type
    data_cleaned['Geography'] = pd.factorize(data_cleaned['Geography'])[0] + 1  # Adding 1 to start numbering from 1
    data_cleaned['Gender'] = pd.factorize(data_cleaned['Gender'])[0] + 1

    # EDA
    if choice == 'eda':
        st.subheader("EDA Section")
        st.text("Exploratory Data Analysis")
        # Preview
        if st.checkbox("Preview Dataset"):
            number = st.number_input("Number to Show",
                                     min_value=1, max_value=100, value=5, step=1)
            st.dataframe(df.head(number))

        # Show columns/ Rows
        if st.button("Column Names"):
            st.write(df.columns)
        # Descriptions
        if st.checkbox("Show Description"):
            st.write(df.describe())
        # Shape
        if st.checkbox("Show Shape of Dataset"):
            st.write(df.shape)
            data_dim = st.radio("Show dimension by", ("Rows", "Columns"))
            if data_dim == 'Rows':
                st.text("Number of Rows")
                st.write(df.shape[0])
            elif data_dim == 'Columns':
                st.text("Number of Columns")
                st.write(df.shape[1])
            else:
                st.write("df.shape")
        # Selection
        if st.checkbox('Select Columns to Show'):
            all_columns = df.columns.tolist()
            selected_columns = st.multiselect("Select Columns", all_columns)
            new_df = df[selected_columns]
            st.dataframe(new_df)
        if st.checkbox("Select Rows to Show"):
            selected_index = st.multiselect("Select Rows", df.head(100).index)
            selected_rows = df.loc[selected_index]
            st.dataframe(selected_rows)
        if st.button("Value Counts"):
            st.text("Value Counts by Has a Credit Card")
            st.write(df.iloc[:, -4].value_counts())
        # Plot
        if st.checkbox('Show Correlation Plot[Matplotlib]'):
            st.set_option('deprecation.showPyplotGlobalUse', False)
            numeric_df = df.select_dtypes(include=[np.number])
            plt.matshow(numeric_df.corr())
            plt.xticks(range(len(numeric_df.columns)), numeric_df.columns, rotation='vertical')
            plt.yticks(range(len(numeric_df.columns)), numeric_df.columns)
            plt.colorbar()
            st.pyplot()

            # plt.matshow(df1.corr())
            # st.pyplot()
        if st.checkbox("Show Correlation Plot[Seaborn]"):
            numeric_columns = df.select_dtypes(include='number')

            # Drop missing values in numeric columns
            numeric_columns = numeric_columns.dropna()

            # Display correlation plot
            st.write(sns.heatmap(numeric_columns.corr(), annot=True))
            st.pyplot()
    elif choice == 'prediction':
        st.subheader("Prediction Section")
        d_geography = {'France': 0, 'Spain': 1, 'Germany': 2, 'other': 3}
        d_gender = {'Female': 0, 'Male': 1}
        credit_score = st.slider("Select Credit Score", 350, 850)
        geography = st.selectbox("Select Location", tuple(d_geography.keys()))
        gender = st.radio("Select Sex", tuple(d_gender.keys()))
        age = st.slider("Select Age", 18, 100)
        tenure = st.slider("Select Tenure", 0, 10)
        balance = st.number_input("Balance", 0, 999999)
        no_products = st.slider("Number of products", 0, 10)
        has_cr_card = int(st.checkbox("Has Credit Card"))
        is_active_member = int(st.checkbox("Is Active Member"))
        salary = st.number_input("Estimate Salary")
        # USER INPUT ENDS HERE

        # GET VALUES FOR EACH INPUT
        k_gender = get_value(gender, d_gender)
        k_geography = get_value(geography, d_geography)

        # RESULT OF USER INPUT
        selected_options = [credit_score, geography, gender, age, tenure, balance, no_products,
                            has_cr_card, is_active_member, salary]
        vectorized_result = [credit_score, k_geography, k_gender, age, tenure, balance, no_products,
                             has_cr_card, is_active_member, salary]
        sample_data = np.array(vectorized_result).reshape(1, -1)

        column_names = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
                        'IsActiveMember', 'EstimatedSalary']

        sample_data_df = pd.DataFrame([vectorized_result], columns=column_names)


        st.info(selected_options)
        st.text("Using this encoding for prediction")
        st.success(vectorized_result)

        if st.button("Predict"):
            model_predictor = load_model_n_predict("churn_modelling_pipeline_v3.pkl")
            prediction = model_predictor.predict(sample_data_df)
            st.success("Predicted Churn as :: {}".format(prediction))

if __name__ == '__main__':
    main()
