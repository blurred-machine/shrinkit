
import streamlit as st
import pandas as pd

class DataFiltration:
    def __init__(self, data):
        self.data = data

    def filter_data(self, status):
        id_column = st.sidebar.selectbox('Choose ID Column', ['none'] + list(self.data.columns))
        target_column = st.sidebar.selectbox('Choose Target Column', self.data.columns)
        print(target_column)

        if target_column:
            status.markdown("#### Status: Target Column Chosen.")
            X = self.data.drop([target_column], axis=1)
            y = self.data.loc[:, [target_column]]
        else:
            status.markdown("#### Status: Choosing Target Columns...")

        
        return X, y, id_column