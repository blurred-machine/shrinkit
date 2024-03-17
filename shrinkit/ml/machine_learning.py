import streamlit as st
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, \
    ElasticNet, OrthogonalMatchingPursuit, BayesianRidge, ARDRegression, HuberRegressor, \
        QuantileRegressor, RANSACRegressor, TheilSenRegressor, PoissonRegressor, GammaRegressor

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier

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
            preds = []
        return preds, data_type
    
    
    def train_model(self, selected_models, models):
        preds = dict()
        print(selected_models)
        for key in selected_models:
            model = models[key]
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            preds[key] = y_pred
        return preds


    def execute_regression_pipeline(self):
        models = ['Linear Regression', 'Ridge', 'Lasso', 'ElasticNet', 'SVM', 'Random Forest Regressor', \
                'Decision Tree Regressor', 'Bayesian Ridge', 'Poisson Regressor', \
                'Gamma Regressor', 'ARD Regression', 'Huber Regressor', \
                'RANSAC Regressor', 'TheilSen Regressor', 'Orthogonal Matching Pursuit']
        
        st.sidebar.markdown("#### Select Regression Models:")
        select_all_toggle = st.sidebar.toggle(label="Select All", key="Select all regression models")
        selected_models = []

        if select_all_toggle:
            selected_models = models
            for model in models:
                st.sidebar.checkbox(model, value=True) 
        else:
            for model in models:
                selected = st.sidebar.checkbox(model)
                if selected:
                    selected_models.append(model)
    
        models = {
                'Linear Regression': LinearRegression(),
                'Ridge' : Ridge(),
                'Lasso' : Lasso(),
                'ElasticNet': ElasticNet(),
                'Decision Tree Regressor': DecisionTreeRegressor(),
                'Bayesian Ridge': BayesianRidge(),
                'Poisson Regressor': PoissonRegressor(),
                'Gamma Regressor': GammaRegressor(),
                'ARD Regression': ARDRegression(),
                'Huber Regressor': HuberRegressor(),
                'RANSAC Regressor': RANSACRegressor(),
                'TheilSen Regressor': TheilSenRegressor(),
                'Orthogonal Matching Pursuit': OrthogonalMatchingPursuit(),
                'SVM': SVR(),
                'Random Forest Regressor': RandomForestRegressor()
            }
        return self.train_model(selected_models, models)


    def execute_classification_pipeline(self):
        # selected_models = st.sidebar.multiselect('Classification Models', ['Logistic Regression', 'SVM', 'KNN', "Random Forest"])
        models = ['Logistic Regression', 'Support Vector', 'KNN', \
                'Decision Tree Classifier', 'Random Forest Classifier', 'MLP Classifier']
        
        st.sidebar.markdown("#### Select Classification Models:")
        select_all_toggle = st.sidebar.toggle(label="Select All", key="Select all classification models")
        selected_models = []

        if select_all_toggle:
            selected_models = models
            for model in models:
                st.sidebar.checkbox(model, value=True) 
        else:
            for model in models:
                selected = st.sidebar.checkbox(model)
                if selected:
                    selected_models.append(model)
        models = {
                'Logistic Regression': LogisticRegression(),
                'Support Vector': SVC(),
                'KNN': KNeighborsClassifier(),
                'Decision Tree Classifier': DecisionTreeClassifier(),
                'Random Forest Classifier': RandomForestClassifier(),
                'MLP Classifier': MLPClassifier()
            }
        return self.train_model(selected_models, models)
        
        
