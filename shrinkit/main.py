import streamlit as st

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report


from ml.imputation import CustomImputer
from ml.filtration import DataFiltration
from ml.machine_learning import MLModeling
from ml.encoding import CustomEncoding
from ml.normalization import CustomNormalizer
from ml.evaluation import CustomEvaluation

import warnings
warnings.filterwarnings("ignore")

class Shrinkit():
    def __init__(self):
        st.markdown("# Shrink-it Playground! 🚀")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            self.data = pd.read_csv(uploaded_file)
            self.is_data_available = True
        else:
            self.is_data_available = False

        

    def run(self):
        status = st.markdown("#### Status: Reading Dataset...")
        st.divider()
        st.sidebar.title('Shrink-it Config ⚙️')
        st.sidebar.write("Choose the configurations below:")
        st.sidebar.divider()
        # ==============================================================
        # =================== READ DATA AND PROCESS IT =================
        # ==============================================================
        st.sidebar.title('Reading Data:')
        c1, c2 = st.sidebar.columns(2)
        show_original_data = c1.toggle(label='Show', key="show original dataset", value=False)
        show_data_profiling = c2.toggle(label='Analysis', key="show profiler results", value=False)
        st.sidebar.write("Note: Keep the Analysis OFF if you are not analyzing the data. It is compute intensive!")
        if show_original_data:
            try:
                st.markdown(f"#### Original Data: {self.data.shape}")
                st.write(self.data.head())
                st.divider()
                self.is_data_available = True
            except:
                self.is_data_available = False

        if self.is_data_available and show_data_profiling:
            st.markdown("### Data Analysis Results & Visualizations:")
            profile = ProfileReport(self.data, title="Pandas Profiling Report", minimal=True, explorative=True)
            st_profile_report(profile, navbar=True)
            # st.write(profile)
        else:
            status.markdown("#### Status: Please upload dataset...")


        if self.is_data_available:
            # ===============================================================
            # ========================= DATA FILTRATION =====================
            # ===============================================================
            st.sidebar.divider()
            st.sidebar.title('Understanding Data:')
            show_filter_results = st.sidebar.toggle(label='Show', key="show data filtered results")
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
            show_imputation_results = st.sidebar.toggle(label='Show', key="missing values show")
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
            show_encoded_results = st.sidebar.toggle(label='Show', key="encoded data show")
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
            show_normalized_results = st.sidebar.toggle(label='Show', key="normalized data show")
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
            train_split = st.sidebar.slider('Training data (%):', 1, 99, 75)
            test_size = (100-train_split)/100
            print(f"Test data size: {test_size}")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100-train_split)/100, random_state=42)

            # ===============================================================
            # ========================= MACHINE LEARNING ====================
            # ===============================================================
            st.sidebar.divider()
            st.sidebar.title('Machine Learning:')
            show_ml_results = st.sidebar.toggle(label='Show Training Results', key="show training metric results")
            c1, c2 = st.sidebar.columns(2)
            show_predictions = c1.toggle(label='Show Predictions', key="show predictions data")
            show_plot = c2.checkbox("Plot")
            ml_modeling = MLModeling(X_train, X_test, y_train, y_test)
            predictions_dict, cateogry = ml_modeling.compute_ML()

            # ======================= EVALUATION METRICS ====================
            eval = CustomEvaluation(y_test, predictions_dict, X_train.shape[1], cateogry)
            matrix_table = eval.evaluate()
            if show_ml_results:
                st.markdown(f"### Evaluation Metrics Table: ({cateogry.capitalize()} Models)")
                st.write(matrix_table)
                st.divider()

            if show_predictions:
                st.markdown(f"### Predictions: ({len(predictions_dict)})")
                st.write(predictions_dict)
                st.divider()

            if show_plot:
                st.line_chart(matrix_table.transpose())
            


if __name__ == "__main__":
    st.divider()
    skt = Shrinkit()
    skt.run()