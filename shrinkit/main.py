import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from imputation import CustomImputer
from filtration import DataFiltration
from machine_learning import MLModeling
from encoding import CustomEncoding
from normalization import CustomNormalizer
from evaluation import CustomEvaluation

import warnings
warnings.filterwarnings("ignore")


class Shrinkit():
    def __init__(self):
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            self.data = pd.read_csv(uploaded_file)
            self.is_data_available = True
        else:
            self.is_data_available = False


    def run(self):
        status = st.markdown("#### Status: Reading Dataset...")
        st.divider()
        # ==============================================================
        # =================== READ DATA AND PROCESS IT =================
        # ==============================================================
        st.sidebar.title('Reading Data:')
        show_original_data = st.sidebar.toggle('Show original data?', value=True)
        if show_original_data:
            try:
                st.markdown(f"#### Original Data: {self.data.shape}")
                st.write(self.data.head())
                st.divider()
                self.is_data_available = True
            except:
                self.is_data_available = False

        if self.is_data_available:
            # ===============================================================
            # ========================= DATA FILTRATION =====================
            # ===============================================================
            st.sidebar.divider()
            st.sidebar.title('Understanding Data:')
            show_filter_results = st.sidebar.toggle('Show filtered data?')
            filter = DataFiltration(self.data)
            X, y, id_column = filter.filter_data(status)
            if show_filter_results:
                st.markdown("### Filtered Data:")
                c1, c2 = st.columns(2)
                c1.write(f"Independent Features: {X.shape}")
                c1.write(X.head(5))
                c2.write(f"Dependent Features: {y.shape}")
                c2.write(y.head(5))
                st.divider()

            print(f"ID COL: {id_column}")
            if id_column and id_column != "none":
                X = X.drop([id_column], axis=1)
            # ===============================================================
            # =================== MISSING VALUES IMPUTATION =================
            # ===============================================================
            st.sidebar.divider()
            st.sidebar.title('Missing Imputation:')
            show_imputation_results = st.sidebar.toggle('Show imputed data?')
            imputer = CustomImputer()
            X = imputer.execute_missing_value_imputation(X, status)
            if show_imputation_results:
                st.markdown(f"### Imputed Dataset: {X.shape}")
                st.write(X)


            # ===============================================================
            # =========================== ENCODING DATA ====================
            # ===============================================================
            st.sidebar.divider()
            st.sidebar.title('Encoding Data:')
            show_encoded_results = st.sidebar.toggle('Show encoded data?')
            encoder = CustomEncoding(X, status)
            X = encoder.encode()
            if show_encoded_results:
                st.markdown(f"### Encoded Dataset: {X.shape}")
                st.write(X)

            # ===============================================================
            # ========================= NORMALIZE DATA ====================
            # ===============================================================
            st.sidebar.divider()
            st.sidebar.title('Normalize Data:')
            show_normalized_results = st.sidebar.toggle('Show normalized data?')
            normalizer = CustomNormalizer(X, status)
            X = normalizer.normalize()
            if show_normalized_results:
                st.markdown(f"### Normalized Dataset: {X.shape}")
                st.write(X)

            # ===============================================================
            # ========================= DATA SPLITTING ======================
            # ===============================================================
            st.sidebar.divider()
            st.sidebar.title('Splitting Data:')
            train_split = st.sidebar.slider('Training data (%):', 0, 100, 75)
            test_size = (100-train_split)/100
            print(f"Test data size: {test_size}")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100-train_split)/100, random_state=42)

            # ===============================================================
            # ========================= MACHINE LEARNING ====================
            # ===============================================================
            st.sidebar.divider()
            st.sidebar.title('Machine Learning:')
            show_ml_results = st.sidebar.toggle('Show ML results?')
            ml_modeling = MLModeling(X_train, X_test, y_train, y_test)
            predictions_dict, cateogry = ml_modeling.compute_ML()

            # ======================= EVALUATION MATRICS ====================
            eval = CustomEvaluation(y_test, predictions_dict, X_train.shape[1], cateogry)
            matrix_table = eval.evaluate()
            if show_ml_results:
                st.markdown(f"### Predictions: ({len(predictions_dict)})")
                st.write(predictions_dict)
                st.write(matrix_table)


if __name__ == "__main__":
    skt = Shrinkit()
    skt.run()