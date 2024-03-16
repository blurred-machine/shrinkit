
import streamlit as st
import pandas as pd

class DataFiltration:
    def __init__(self, data):
        self.data = data

    def filter_data(self, status):
        id_column = st.sidebar.selectbox('Choose ID Column', ['none'] + list(self.data.columns))
        target_columns = st.sidebar.multiselect('Choose Target Columns', self.data.columns)
        print(target_columns)

        if len(target_columns) >= 0:
            status.markdown("#### Status: Target Column Chosen.")
            X = self.data.drop(target_columns, axis=1)
            y = self.data.loc[:, target_columns]
        else:
            status.markdown("#### Status: Choosing Target Columns...")

        
        return X, y, id_column