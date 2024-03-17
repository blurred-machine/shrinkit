from sklearn.impute import SimpleImputer
import streamlit as st
import pandas as pd
import numpy as np

class CustomImputer:
    def __init__(self):
        pass

    def execute_missing_value_imputation(self, X, status):
        
        c1, c2 = st.sidebar.columns(2)
        imputation_strategy_num = c1.selectbox('Numerical', ['none', 'mean', 'median', 'most_frequent', 'constant'])
        imputation_strategy_cat = c2.selectbox('Categorical', ['none', 'most_frequent', 'constant'])

        if imputation_strategy_num == 'constant':
            const_fill_num = c1.text_input('Numerical Fill Value', -1)
        else:
            const_fill_num = None
        
        if imputation_strategy_cat == 'constant':
            const_fill_cat = c2.text_input('Categorical Fill Value', "missing")
        else:
            const_fill_cat = None

        num_data = X.select_dtypes(include='number')
        print(f"Num cols shape: {len(num_data.columns)}")

        cat_cols = [col for col in X.columns if col not in num_data.columns]
        print(f"Cat cols shape: {len(cat_cols)}")
        cat_data = X.loc[:, cat_cols]

        if imputation_strategy_num != "none":
            status.markdown("#### Status: Missing Values Imputed.")
            if const_fill_num:
                imp_num = SimpleImputer(missing_values=np.nan, strategy=imputation_strategy_num, fill_value=int(const_fill_num))
            else:
                imp_num = SimpleImputer(missing_values=np.nan, strategy=imputation_strategy_num)
            num_data_imputed = pd.DataFrame(imp_num.fit_transform(num_data), columns=num_data.columns)
        else:
            num_data_imputed = num_data
            status.markdown("#### Status: Imputing missing values...")
            
        if imputation_strategy_cat != "none":
            status.markdown("#### Status: Missing Values Imputed.")
            imp_cat = SimpleImputer(missing_values=np.nan, strategy=imputation_strategy_cat, fill_value=const_fill_cat)
            cat_data_imputed = pd.DataFrame(imp_cat.fit_transform(cat_data), columns=cat_data.columns)
        else:
            cat_data_imputed = cat_data
            status.markdown("#### Status: Imputing missing values...")
        return pd.concat([num_data_imputed, cat_data_imputed], axis=1)