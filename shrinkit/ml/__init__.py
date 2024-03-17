import streamlit as st
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

from shrinkit.ml.imputation import *
from shrinkit.ml.filtration import *
from shrinkit.ml.machine_learning import *
from shrinkit.ml.encoding import *
from shrinkit.ml.normalization import *
from shrinkit.ml.evaluation import *