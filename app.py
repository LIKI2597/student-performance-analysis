import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from math import ceil
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Set up Streamlit app
st.title("Student Performance Analysis using Streamlit")
st.image("download stu.png", width=450)
st.title("Case Study on Student Performance")

# Use Pandas to load the custom CSV file
data = pd.read_csv("StudentsPerformance.csv")  # Make sure the path to the CSV is correct

# Add new columns 'total_score' and 'percentage' to the dataset
data['total_score'] = data['math score'] + data['reading score'] + data['writing score']
data['percentage'] = data['total_score'] / 3

# Apply ceil to the percentage values
for i in range(len(data)):
    data['percentage'][i] = ceil(data['percentage'][i])

# Label encoding for categorical variables
le = LabelEncoder()
data['test preparation course'] = le.fit_transform(data['test preparation course'])
data['lunch'] = le.fit_transform(data['lunch'])
data['gender'] = le.fit_transform(data['gender'])
data['parental level of education'] = le.fit_transform(data['parental level of education'])

# Label encoding for race/ethnicity groups
data['race/ethnicity'] = data['race/ethnicity'].replace({
    'group A': 1, 'group B': 2, 'group C': 3, 'group D': 4, 'group E': 5
})

# Add pass/fail columns based on a passing threshold
passing_threshold = 40  # Adjust as necessary
data['pass_math'] = data['math score'].apply(lambda x: 'Pass' if x >= passing_threshold else 'Fail')
data['pass_reading'] = data['reading score'].apply(lambda x: 'Pass' if x >= passing_threshold else 'Fail')
data['pass_writing'] = data['writing score'].apply(lambda x: 'Pass' if x >= passing_threshold else 'Fail')

# Label encoding for pass/fail columns
data['pass_math'] = le.fit_transform(data['pass_math'])
data['pass_reading'] = le.fit_transform(data['pass_reading'])
data['pass_writing'] = le.fit_transform(data['pass_writing'])

# Features and target variable
x = data.iloc[:, :14]  # Independent variables (all columns except target)
y = data['pass_math']  # Dependent variable (you can change to pass_reading or pass_writing)

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=45)

# Scaling numerical data
numerical_cols = x_train.select_dtypes(include=['number']).columns
mm = MinMaxScaler()
x_train_scaled = mm.fit_transform(x_train[numerical_cols])
x_test_scaled = mm.transform(x_test[numerical_cols])

# One-hot encoding for categorical columns
categorical_cols = x_train.select_dtypes(include=['object']).columns
ohe = OneHotEncoder(handle_unknown='ignore')
x_train_ohe = ohe.fit_transform(x_train[categorical_cols]).toarray()
x_test_ohe = ohe.transform(x_test[categorical_cols]).toarray()

# Combine scaled numerical and encoded categorical data
x_train_final = np.concatenate([x_train_scaled, x_train_ohe], axis=1)
x_test_final = np.concatenate([x_test_scaled, x_test_ohe], axis=1)

# Sidebar menu for different options
menu = st.sidebar.radio("Menu", ["Home", "Analysis"])

if menu == "Home":
    st.image("images stud.jpeg")
    st.header("Tabular Data of Students")
    if st.checkbox("Tabular Data"):
        st.table(data.head(150))
    st.header("Statistical Summary")
    if st.checkbox("Statistics"):
        st.table(data.describe())
    st.header("Correlation Graph")
    numeric_data = data.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

if menu == "Analysis":
    st.title("Machine Learning Models")
    
    model_choice = st.selectbox("Choose a model", ["Logistic Regression", "Random Forest Classifier"])

    if model_choice == "Logistic Regression":
        # Logistic Regression Model
        model = LogisticRegression()
        model.fit(x_train_final, y_train)
        y_pred = model.predict(x_test_final)

        # Display results
        st.write("### Logistic Regression Results")
        st.write("Training Accuracy: ", model.score(x_train_final, y_train))
        st.write("Testing Accuracy: ", model.score(x_test_final, y_test))

    elif model_choice == "Random Forest Classifier":
        # Random Forest Classifier
        model = RandomForestClassifier()
        model.fit(x_train_final, y_train)
        y_pred = model.predict(x_test_final)

        # Display results
        st.write("### Random Forest Classifier Results")
        st.write("Training Accuracy: ", model.score(x_train_final, y_train))
        st.write("Testing Accuracy: ", model.score(x_test_final, y_test))

    st.title("Graphs")
    graph = st.selectbox("Different types of Graphs", [
        "Scatter plot", "Bar Graph", "Histogram", "Math Score Count Plot", 
        "Total Score Distribution", "Percentage Distribution", 
        "Parental Education", "Race/Ethnicity Comparison", 
        "Gender Comparison"
    ])

    # Your previous graph plotting code goes here...
