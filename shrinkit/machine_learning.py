import streamlit as st
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline



class MLModeling:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


    def compute_ML(self):
        data_type = st.sidebar.selectbox('ML Model Type', ['none', 'regression', 'classification'])
        if data_type == "regression":
            preds = self.execute_regression_pipeline()
        elif data_type == "classification":
            preds = self.execute_classification_pipeline()
        else:
            st.write("Choose an ML Problem category!")
            preds = []
        return preds, data_type
    
    
    def train_model(self, selected_models, models):
        preds = dict()
        for key in selected_models:
            model = models[key]
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            preds[key] = y_pred
        return preds


    def execute_regression_pipeline(self):
        selected_models = st.sidebar.multiselect('Regression Models', ['Linear Regression', 'SVM', 'Random Forest'])
        models = {
                'Linear Regression': LinearRegression(),
                'SVM': SVR(),
                'Random Forest': RandomForestRegressor()
            }
        return self.train_model(selected_models, models)


    def execute_classification_pipeline(self):
        selected_models = st.sidebar.multiselect('Classification Models', ['Logistic Regression', 'SVM', 'KNN', "Random Forest"])
        models = {
                'Logistic Regression': LogisticRegression(),
                'SVM': SVC(),
                'KNN': KNeighborsClassifier(),
                'Random Forest': RandomForestClassifier()
            }
        return self.train_model(selected_models, models)
        
        
