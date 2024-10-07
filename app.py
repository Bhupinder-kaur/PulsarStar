import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import base64

# Load dataset
file_path = (r'S:\Data Science Prakash Senapati\AllDatasets\pulsar_stars.csv')  # Replace with the correct file path
df = pd.read_csv(file_path)

# Feature and target split
X = df.drop('target_class', axis=1)
y = df['target_class']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to load a model from file
def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model


# Title for the app
st.markdown("<h1 style='text-align: center;'>Pulsar Star Classification</h1>", unsafe_allow_html=True)

# Main content for model selection and inputs
st.subheader("Choose a model for prediction")
model_option = st.selectbox("Select Model", 
                            ("SVM - RBF Kernel and C = 100", 
                             "SVM - RBF Kernel and C = 1000", 
                             "SVM - Linear Kernel and C = 1.0", 
                             "SVM - Linear Kernel and C = 100.0", 
                             "SVM - Linear Kernel and C = 1000.0", 
                             "SVM - Polynomial Kernel and C = 1.0", 
                             "SVM - Polynomial Kernel and C = 100.0", 
                             "SVM - Sigmoid Kernel and C = 1.0", 
                             "SVM - Sigmoid Kernel and C = 100.0", 
                             "SVM - Default Model"))

# Inputs for user
st.subheader("Input Features")
IP_mean = st.number_input("Mean of the integrated profile", value=0.0)
IP_sd = st.number_input("Standard deviation of the integrated profile", value=0.0)
IP_kurtosis = st.number_input("Excess kurtosis of the integrated profile", value=0.0)
IP_skewness = st.number_input("Skewness of the integrated profile", value=0.0)
DMSNR_mean = st.number_input("Mean of the DM-SNR curve", value=0.0)
DMSNR_sd = st.number_input("Standard deviation of the DM-SNR curve", value=0.0)
DMSNR_kurtosis = st.number_input("Excess kurtosis of the DM-SNR curve", value=0.0)
DMSNR_skewness = st.number_input("Skewness of the DM-SNR curve", value=0.0)

# Combine all inputs into a feature vector
input_data = [[IP_mean, IP_sd, IP_kurtosis, IP_skewness, DMSNR_mean, DMSNR_sd, DMSNR_kurtosis, DMSNR_skewness]]

# Scale input data
input_data_scaled = scaler.transform(input_data)

# Model predictions
if st.button("Predict"):
    # Load the appropriate model based on the user's selection
    if model_option == "SVM - RBF Kernel and C = 100":
        model = load_model('svc_default_model.pkl')
    elif model_option == "svc_rbf_100_model.pkl":
        model = load_model('svc_rbf_100_model.pkl')
    elif model_option == "SVM - Linear Kernel and C = 1.0":
        model = load_model('svc_linear_1_model.pkl')
    elif model_option == "SVM - Linear Kernel and C = 100.0":
        model = load_model('svc_linear_100_model.pkl')
    elif model_option == "SVM - Linear Kernel and C = 1000.0":
        model = load_model('svc_linear_1000_model.pkl')
    elif model_option == "SVM - Polynomial Kernel and C = 1.0":
        model = load_model('poly_svc_1_model.pkl')
    elif model_option == "SVM - Polynomial Kernel and C = 100.0":
        model = load_model('poly_svc_100_model.pkl')
    elif model_option == "SVM - Sigmoid Kernel and C = 1.0":
        model = load_model('svc_sigmoid_1_model.pkl')
    elif model_option == "SVM - Sigmoid Kernel and C = 100.0":
        model = load_model('svc_sigmoid_100_model.pkl')
 
    # Perform prediction
    prediction = model.predict(input_data_scaled)
    
    if prediction[0] == 1:
        st.success("The star is likely a pulsar.")
    else:
        st.warning("The star is likely not a pulsar.")
