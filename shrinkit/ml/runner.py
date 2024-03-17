import sys
import os
import streamlit.web.cli as stcli
import subprocess


class ShrinkitRunner:
    def __init__(self):
        pass

    def streamlit_run(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(this_dir)
        sys.path.append(this_dir)
        sys.argv = ["streamlit", "run", "main.py", "--global.developmentMode=false", "--browser.gatherUsageStats=false"]
        sys.exit(stcli.main())

    # def run(self):
        # sys.argv = ["streamlit", "run", "./shrinkit/main.py"]
        # sys.exit(stcli.main())

        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # print(current_dir)
        # main_file_path = os.path.join(current_dir, 'main.py')
        # sys.argv = ["streamlit", "run", main_file_path]
        # sys.exit(stcli.main())
        
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # subprocess.Popen(f'sh {current_dir}/top_run.sh', shell=True)
