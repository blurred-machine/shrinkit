import sys
import streamlit.web.cli as stcli

class ShrinkitRunner:
    def __init__(self):
        pass

    def run(self):
        sys.argv = ["streamlit", "run", "shrinkit/main.py"]
        sys.exit(stcli.main())