import streamlit as st
st.title('Brain Stroke Prevention')


import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

st.title('Brain Stroke Prediction')

# st.write('Please select the data the following limitations:')
# st.write('For Gender : Female = 0, Male = 1 and Other = 2')
# st.text('For Age : From 0 to 82 years')
# st.text('For Hypertension : From 0 to 1')
# st.text('For Heart Disease : From 0 to 1')
# st.text('For Ever Married : No = 0 and Yes = 1')
# st.text('For Work Type : Government Job = 0, Never Worked = 1, Private = 2, Self-employed = 3, Children = 4')
# st.text('For Residence Type : Rural = 0, Urban = 1')
# st.text('For Average Glucose Level : From 55 to 272')
# st.text('For Blood Pressure : Fro 10 to 98')
# st.text('For Smoking Status : Unknown = 0, Formely Smoked = 1, Never Smoked = 2 and Smokes = 3')
# st.text('The index number "0" express "Brain Stroke".')
# st.text('The index number "1" express "Not Brain Stroke".')

Brain = pd.read_csv('healthcare-dataset-stroke-data.csv')
classifier = st.sidebar.selectbox('Choose model classifier', ('Logistic Regression', 'Naive Bayes', 'K-Nearest Neighbors', 'Support Vector Machine', 'Decision Tree Classifier', 'Random Forest Classifier', 'XGB Classifier'))

Brain.dropna(inplace = True)

# '''Under Sampling'''
NoStroke = Brain[Brain.stroke == 0]
Stroke = Brain[Brain.stroke == 1]
NoStrokeSample = NoStroke.sample(n = 50)
Brain = pd.concat([NoStrokeSample, Stroke], axis = 0)
NoStrokeSample = NoStroke.sample(n = 50)

# '''For Checkbox'''
CheckData = st.sidebar.checkbox('Brain Stroke Data')
CheckMetrics = st.sidebar.checkbox('Predict Test')

if CheckData:
    st.write(Brain)
    
# '''User Input'''
def InputUser():
    col1, col2 = st.columns(2)
    with col1:
        Gender = st.number_input('Gender: Female = 0, Male = 1', min_value = 0, max_value = 1, value = 0, step = 1)
    with col2:
        Age = st.number_input('Age: Please enter from o to 82', min_value = 0, max_value = 82, value = 45, step = 1)
        
    col1, col2 = st.columns(2)        
    with col1:
        Hypertension = st.number_input('Hypertension: No = 0, Yes = 1', min_value = 0, max_value = 1, value = 1, step = 1)
    with col2:
        HeartDisease = st.number_input('Heart Disease: No = 0, Yes = 1', min_value = 0, max_value = 1, value = 0, step = 1)
    
    col1, col2 = st.columns(2)    
    with col1:
        Ever_Married = st.number_input('Ever Married: No = 0, Yes = 1', min_value = 0, max_value = 1, value = 1, step = 1)
    with col2:
        Work_Type = st.number_input('Work Type: Government Job = 0, Never Worked = 1, Private = 2, Self-employed = 3', min_value = 0, max_value = 4, value = 2, step = 1)
        
    col1, col2 = st.columns(2)        
    with col1:
        Residence = st.number_input('Residence: Rural = 0, Urban = 1', min_value = 0, max_value = 1, value = 1, step = 1)
    with col2:
        Avg_Glu_Level = st.number_input('Average Glucose Level: Please enter from 55 to 272', min_value = 55, max_value = 272, value = 150, step = 1)
        
    col1, col2 = st.columns(2)        
    with col1:
        BMI = st.number_input('Blood Pressure: Please enter from 10 to 98', min_value = 10, max_value = 98, value = 72, step = 1)
    with col2:
        Smoking_Status = st.number_input('Smoking Status: Unknown = 0, Formely Smoked = 1, Never Smoked = 2, Smokes = 3', min_value = 0, max_value = 3, value = 2)
    
        
    Data = {'Gender' : Gender, 
            'Hypertension' : Hypertension, 
            'HeartDisease' : HeartDisease, 
            'Ever_Married' : Ever_Married, 
            'Work_Type' : Work_Type,
            'Residence' : Residence, 
            'Avg_Glu_Level' : Avg_Glu_Level, 
            'BMI' : BMI, 
            'Smoking_Status' : Smoking_Status,
            'Age' : Age}
    features = pd.DataFrame(Data, index = [0])
    return features

encoder = LabelEncoder()
Brain['gender'] = encoder.fit_transform(Brain['gender'])
gender = {index : label for index, label in enumerate(encoder.classes_)}
#gender

Brain['ever_married'] = encoder.fit_transform(Brain['ever_married'])
ever_married = {index : label for index, label in enumerate(encoder.classes_)}
#ever_married

Brain['work_type'] = encoder.fit_transform(Brain['work_type'])
work_type = {index : label for index, label in enumerate(encoder.classes_)}
#work_type

Brain['Residence_type'] = encoder.fit_transform(Brain['Residence_type'])
Residence_type = {index : label for index, label in enumerate(encoder.classes_)}
#Residence_type

Brain['smoking_status'] = encoder.fit_transform(Brain['smoking_status'])
smoking_status = {index : label for index, label in enumerate(encoder.classes_)}
#smoking_status

#Brain

# '''Dividing Feature and Label'''
x = Brain.iloc[: , 1 :11]
y = Brain.iloc[: , 11 : 12]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# '''Modeling'''
if classifier == 'Logistic Regression':
    st.subheader('Using Logistic Regression Classifier')
    from sklearn.linear_model import LogisticRegression
    LR_model = LogisticRegression()
    LR_model.fit(x_train, y_train)
    
# '''Evaluation'''
    x_train_pred = LR_model.predict(x_train)
    x_test_pred = LR_model.predict(x_test)
    
    input_df = InputUser()
    test_button = st.button('Predict Test')
    if test_button:
        input_data_as_numpy_array = np.asarray(input_df)
        input_data_reshaped = input_data_as_numpy_array.reshape(1 , -1)
        std_data =scaler.transform(input_data_reshaped)
        
        prediction = LR_model.predict(std_data)
        if prediction[0] == 0:
            st.success('This person is at risk of brain stroke.')
        if prediction[0] == 1:
            st.success('This person is not at risk of brain stroke.')
            
# '''Naive Bayes'''         
if classifier == 'Naive Bayes':
        st.subheader('Using Naive Bayes Classifier')
        from sklearn.naive_bayes import GaussianNB
        NB_model = GaussianNB()
        NB_model.fit(x_train, y_train)
        
        x_train_pred = NB_model.predict(x_train)
        x_test_pred = NB_model.predict(x_test)
    
        input_df = InputUser()
        test_button = st.button('Predict Test')
        if test_button:
            input_data_as_numpy_array = np.asarray(input_df)
            input_data_reshaped = input_data_as_numpy_array.reshape(1 , -1)
            std_data =scaler.transform(input_data_reshaped)
        
            prediction = NB_model.predict(std_data)
            if prediction[0] == 0:
                st.success('This person is at risk of brain stroke.')
            if prediction[0] == 1:
                st.success('This person is not at risk of brain stroke.')
            
# '''K-Nearest Neighbors'''         
if classifier == 'K-Nearest Neighbors':
        st.subheader('Using K-Nearest Neighbors Classifier')
        from sklearn.neighbors import KNeighborsClassifier
        KNN_model = KNeighborsClassifier(n_neighbors = 13)
        KNN_model.fit(x_train, y_train)
    
        x_train_pred = KNN_model.predict(x_train)
        x_test_pred = KNN_model.predict(x_test)
    
        input_df = InputUser()
        test_button = st.button('Predict Test')
        if test_button:
            input_data_as_numpy_array = np.asarray(input_df)
            input_data_reshaped = input_data_as_numpy_array.reshape(1 , -1)
            std_data =scaler.transform(input_data_reshaped)
        
            prediction = KNN_model.predict(std_data)
            if prediction[0] == 0:
                st.success('This person is at risk of brain stroke.')
            if prediction[0] == 1:
                st.success('This person is not at risk of brain stroke.')
    
# '''Support Vector Machine'''
if classifier == 'Support Vector Machine':
        st.subheader('Using Support Vector Machine Classifier')
        from sklearn.svm import SVC
        SVM_model = SVC(gamma = 'auto')
        SVM_model.fit(x_train, y_train)
    
        x_train_pred = SVM_model.predict(x_train)
        x_test_pred = SVM_model.predict(x_test)
    
        input_df = InputUser()
        test_button = st.button('Predict Test')
        if test_button:
            input_data_as_numpy_array = np.asarray(input_df)
            input_data_reshaped = input_data_as_numpy_array.reshape(1 , -1)
            std_data =scaler.transform(input_data_reshaped)
        
            prediction = SVM_model.predict(std_data)
            if prediction[0] == 0:
                st.success('This person is at risk of brain stroke.')
            if prediction[0] == 1:
                st.success('This person is not at risk of brain stroke.')
    
        #'''Decision Tree Classifier'''         
if classifier == 'Decision Tree Classifier':
    st.subheader('Using Decision Tree Classifier')
    from sklearn import tree 
    DTC_model = tree.DecisionTreeClassifier()
    DTC_model.fit(x_train, y_train)
    
    x_train_pred = DTC_model.predict(x_train)
    x_test_pred = DTC_model.predict(x_test)
    
    input_df = InputUser()
    test_button = st.button('Predict Test')
    if test_button:
        input_data_as_numpy_array = np.asarray(input_df)
        input_data_reshaped = input_data_as_numpy_array.reshape(1 , -1)
        std_data =scaler.transform(input_data_reshaped)
        
        prediction = DTC_model.predict(std_data)
        if prediction[0] == 0:
            st.success('This person is at risk of brain stroke.')
        if prediction[0] == 1:
            st.success('This person is not at risk of brain stroke.')
    
        #'''Random Forest Classifier'''         
if classifier == 'Random Forest Classifier':
    st.subheader('Using Random Forest Classifier')
    from sklearn.ensemble import RandomForestClassifier 
    RFC_model = RandomForestClassifier(n_estimators = 10)
    RFC_model.fit(x_train, y_train)
    
    x_train_pred = RFC_model.predict(x_train)
    x_test_pred = RFC_model.predict(x_test)
    
    input_df = InputUser()
    test_button = st.button('Predict Test')
    if test_button:
        input_data_as_numpy_array = np.asarray(input_df)
        input_data_reshaped = input_data_as_numpy_array.reshape(1 , -1)
        std_data =scaler.transform(input_data_reshaped)
        
        prediction = RFC_model.predict(std_data)
        if prediction[0] == 0:
            st.success('This person is at risk of brain stroke.')
        if prediction[0] == 1:
            st.success('This person is not at risk of brain stroke.')
    
        #'''XGB Classifier'''         
if classifier == 'XGB Classifier':
    st.subheader('Using XGB Classifier')
    from xgboost import XGBClassifier
    XGB_model = XGBClassifier(objective='binary:logistic', learning_rate = 0.2)
    xGB_model.fit(x_train, y_train)
    
    x_train_pred = XGB_model.predict(x_train)
    x_test_pred = XGB_model.predict(x_test)
    
    input_df = InputUser()
    test_button = st.button('Predict Test')
    if test_button:
        input_data_as_numpy_array = np.asarray(input_df)
        input_data_reshaped = input_data_as_numpy_array.reshape(1 , -1)
        std_data =scaler.transform(input_data_reshaped)
        
        prediction = XGB_model.predict(std_data)
        if prediction[0] == 0:
            st.success('This person is at risk of brain stroke.')
        if prediction[0] == 1:
            st.success('This person is not at risk of brain stroke.')
