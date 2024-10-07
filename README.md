# PulsarStar

🌟 Pulsar Star Classification Using SVM Models
Welcome to my Pulsar Star Classification project! 🚀

📂 Project Overview
This project focuses on classifying pulsar stars from non-pulsars using Support Vector Machines (SVM) with various kernel functions and hyperparameters. Pulsars are rotating neutron stars that emit beams of electromagnetic radiation, and detecting them is a crucial task in astrophysics.

The dataset used contains attributes from pulsar candidates, derived from integrated pulse profiles and DM-SNR (dispersion measure signal-to-noise ratio) curves. This project implements multiple SVM models, allowing you to select the kernel type and C parameter, which adjusts the decision boundary for optimal classification.

⚙️ Features
Multiple SVM Models: RBF, Linear, Polynomial, and Sigmoid Kernels with varying C values.
Feature Scaling: Preprocessing the data using StandardScaler for effective SVM training.
User Input: Enter integrated profile and DM-SNR statistics to classify stars as pulsars or not.
Real-time Prediction: Make predictions based on user input using the chosen model.
Streamlit Interface: The app provides an interactive UI for model selection and data input.
💻 Technology Stack
Python: Core language
Scikit-Learn: For model training and evaluation
Pandas: For data manipulation
Streamlit: To build an interactive web app
Pickle: For saving and loading trained models
🚀 How to Run
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/pulsar-classification
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy code
streamlit run app.py
In the browser, enter the features (integrated profile stats and DM-SNR stats) and select your preferred SVM model to predict if the star is a pulsar.

📊 Dataset
The dataset used for this project is sourced from UCI Machine Learning Repository. It contains features from both pulsar and non-pulsar stars based on their integrated profiles and DM-SNR curves.

🔧 SVM Models Implemented
RBF Kernel: C = 100, C = 1000
Linear Kernel: C = 1.0, C = 100.0, C = 1000.0
Polynomial Kernel: C = 1.0, C = 100.0
Sigmoid Kernel: C = 1.0, C = 100.0
🎯 Goal
The primary goal of this project is to build a flexible and interactive classification system to distinguish pulsar stars from non-pulsars. This tool is useful for astronomers and researchers in identifying these stellar objects in large datasets.

🤝 Contributing
Feel free to fork this project, contribute, and submit PRs! Contributions are welcome to improve the classification accuracy, extend the app functionality, or add more machine learning models.

