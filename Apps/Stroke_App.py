import time

import streamlit as st
import pandas as pd
from Classifier_Models import Classifier_model_builder_stroke as cmb
import pickle
import numpy as np

def app():
    st.write("""
    # Stroke Detector

    This app predicts whether a person have chances for stroke or not

    """)

    st.sidebar.header('User Input Features')
    # st.sidebar.markdown("""
    # [Import input CSV file](https://github.com/ChakraDeep8/Heart-Disease-Detector/tree/master/res)""")

    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def patient_details():
            sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
            age = st.sidebar.slider('Age', 0, 103)
            hypertension = st.sidebar.selectbox('Hypertension', ['Yes', 'No'])
            heart_disease = st.sidebar.selectbox('Heart Disease', ['Yes', 'No'])
            marrige_status = st.sidebar.selectbox('Marraige Status', ['Yes', 'No'])
            work_type = st.sidebar.selectbox('Work Type',
                                             ['Never Worked', 'Children', 'Government Job', 'Self-Employed', 'Private'])
            residence_type = st.sidebar.selectbox('Residence Type', ['Urban', 'Rural'])
            glucose_level = st.sidebar.slider('Glucose level', 50, 272)

            bmi = st.sidebar.slider('BMI', 10, 100)
            smoking_status = st.sidebar.selectbox('Smoking status', ['Yes', 'No'])

            data = {'age': age,
                    'sex': sex,
                    'hypertension': hypertension,
                    'heart_disease': heart_disease,
                    'ever_married': marrige_status,
                    'work_type': work_type,
                    'Residence_type': residence_type,
                    'avg_glucose_level': glucose_level,
                    'bmi': bmi,
                    'smoking_status': smoking_status,
                    }

            features = pd.DataFrame(data, index=[0])
            return features

        input_df = patient_details()

    stroke_disease_raw = pd.read_csv('res/stroke_data.csv')
    stroke = stroke_disease_raw.drop(columns=['stroke'])
    df = pd.concat([input_df, stroke], axis=0)

    # Encoding of ordinal features
    encode = ['sex', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    for col in encode:
        dummy = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummy], axis=1)
        del df[col]
    df = df[:1]  # Selects only the first row (the user input data)
    df.loc[:, ~df.columns.duplicated()]

    if uploaded_file is not None:
        st.write(df)
    else:
        st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
        df = df.loc[:, ~df.columns.duplicated()]
        st.write(df)

    # Load the classification models
    load_clf_NB = pickle.load(open('res/stroke_disease_classifier_NB.pkl', 'rb'))
    load_clf_KNN = pickle.load(open('res/stroke_disease_classifier_KNN.pkl', 'rb'))
    load_clf_DT = pickle.load(open('res/stroke_disease_classifier_DT.pkl', 'rb'))
    load_clf_LR = pickle.load(open('res/stroke_disease_classifier_LR.pkl', 'rb'))
    load_clf_RF = pickle.load(open('res/stroke_disease_classifier_RF.pkl', 'rb'))

    # Apply models to make predictions
    prediction_NB = load_clf_NB.predict(df)
    prediction_proba_NB = load_clf_NB.predict_proba(df)
    prediction_KNN = load_clf_KNN.predict(df)
    prediction_proba_KNN = load_clf_KNN.predict_proba(df)
    prediction_DT = load_clf_DT.predict(df)
    prediction_proba_DT = load_clf_DT.predict_proba(df)
    prediction_LR = load_clf_LR.predict(df)
    prediction_proba_LR = load_clf_LR.predict_proba(df)
    prediction_RF = load_clf_RF.predict(df)
    prediction_proba_RF = load_clf_RF.predict_proba(df)

    def NB():
        st.subheader('Naive Bayes Prediction')
        NB_prediction = np.array([0, 1])
        if NB_prediction[prediction_NB] == 1:
            st.write("<p style='font-size:20px;color: orange'><b>You are getting stroke.</b></p>",
                     unsafe_allow_html=True)
        else:
            st.write("<p style='font-size:20px;color: green'><b>You are fine.</b></p>", unsafe_allow_html=True)
        st.subheader('Naive Bayes Prediction Probability')
        st.write(prediction_proba_NB)
        cmb.plt_NB()

    def KNN():
        st.subheader('K-Nearest Neighbour Prediction')
        knn_prediction = np.array([0, 1])
        if knn_prediction[prediction_KNN] == 1:
            st.write("<p style='font-size:20px;color: orange'><b>You are getting stroke.</b></p>",
                     unsafe_allow_html=True)
        else:
            st.write("<p style='font-size:20px;color: green'><b>You are fine.</b></p>", unsafe_allow_html=True)
        st.subheader('KNN Prediction Probability')
        st.write(prediction_proba_KNN)
        cmb.plt_KNN()

    def DT():
        st.subheader('Decision Tree Prediction')
        DT_prediction = np.array([0, 1])
        if DT_prediction[prediction_DT] == 1:
            st.write("<p style='font-size:20px; color: orange'><b>You are getting stroke.</b></p>",
                     unsafe_allow_html=True)
        else:
            st.write("<p style='font-size:20px;color: green'><b>You are fine.</b></p>", unsafe_allow_html=True)
        st.subheader('Decision Tree Prediction Probability')
        st.write(prediction_proba_DT)
        cmb.plt_DT()

    def LR():
        st.subheader('Logistic Regression Prediction')
        DT_prediction = np.array([0, 1])
        if DT_prediction[prediction_LR] == 1:
            st.write("<p style='font-size:20px; color: orange'><b>You are getting stroke.<b></p>",
                     unsafe_allow_html=True)
        else:
            st.write("<p style='font-size:20px;color: green'><b>You are fine.</b></p>", unsafe_allow_html=True)
        st.subheader('Logistic Regression Probability')
        st.write(prediction_proba_LR)
        cmb.plt_LR()

    def RF():
        st.subheader('Random Forest Prediction')
        DT_prediction = np.array([0, 1])
        if DT_prediction[prediction_RF] == 1:
            st.write("<p style='font-size:20px; color: orange'><b>You are getting stroke.</b></p>",
                     unsafe_allow_html=True)
        else:
            st.write("<p style='font-size:20px;color: green'><b>You are fine.</b></p>", unsafe_allow_html=True)
        st.subheader('Random Forest Probability')
        st.write(prediction_proba_RF)
        cmb.plt_LR()

    def predict_best_algorithm():
        if cmb.best_model == 'Naive Bayes':
            st.write("<p style='font-size:24px;'>Best Algorithm: Naive Bayes</p>", unsafe_allow_html=True)
            NB()

        elif cmb.best_model == 'K-Nearest Neighbors (KNN)':
            st.write("<p style='font-size:24px;'>Best Algorithm: K-Nearest Neighbour</p>", unsafe_allow_html=True)
            KNN()

        elif cmb.best_model == 'Decision Tree':
            st.write("<p style='font-size:24px;'>Best Algorithm: Decision Tree</p>", unsafe_allow_html=True)
            DT()

        elif cmb.best_model == 'Logistic Regression':
            st.write("<p style='font-size:24px;'>Best Algorithm: Logistic Regression</p>", unsafe_allow_html=True)
            LR()

        elif cmb.best_model == 'Random Forest':
            st.write("<p style='font-size:24px;'>Best Algorithm: Random Forest</p>", unsafe_allow_html=True)
            RF()
        else:
            st.write("<p style='font-size:20px;color: green'><b>You are fine.</b></p>", unsafe_allow_html=True)

    # Displays the user input features

    with st.expander("Prediction Results"):
        # Display the input dataframe
        st.write("Your input values are shown below:")
        st.dataframe(input_df)
        # Call the predict_best_algorithm() function
        predict_best_algorithm()

    # Create a multiselect for all the plot options
    selected_plots = st.multiselect("Select plots to display",
                                    ["Naive Bayes", "K-Nearest Neighbors", "Decision Tree", "Logistic Regression",
                                     "Random Forest"])

    # Check the selected plots and call the corresponding plot functions

    placeholder = st.empty()

    # Check the selected plots and call the corresponding plot functions
    if "Naive Bayes" in selected_plots:
        with st.spinner("Generating Naive Bayes...."):
            cmb.plt_NB()
            time.sleep(1)

    if "K-Nearest Neighbors" in selected_plots:
        with st.spinner("Generating KNN...."):
            cmb.plt_KNN()
            time.sleep(1)

    if "Decision Tree" in selected_plots:
        with st.spinner("Generating Decision Tree...."):
            cmb.plt_DT()
            time.sleep(1)

    if "Logistic Regression" in selected_plots:
        with st.spinner("Generating Logistic Regression...."):
            cmb.plt_LR()
            time.sleep(1)

    if "Random Forest" in selected_plots:
        with st.spinner("Generating Random Forest...."):
            cmb.plt_RF()
            time.sleep(1)

    # Remove the placeholder to display the list options
    placeholder.empty()