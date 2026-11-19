"""HUVEC Damage Analyzer - start with ``start.bat`` (Windows) or ``./start.sh``.

Everything runs on this computer; images never leave it.
"""

import streamlit as st

st.set_page_config(page_title="HUVEC Damage Analyzer", page_icon=":material/biotech:", layout="wide")

from ui import common  # noqa: E402  (after set_page_config)

common.init_page()

pages = {
    "": [st.Page("ui/page_home.py", title="Home", icon=":material/home:", default=True)],
    "Analyze images": [
        st.Page("ui/page_analyze.py", url_path="analyze", title="1 · Analyze images", icon=":material/upload_file:"),
        st.Page("ui/page_review.py", url_path="review", title="2 · QC review", icon=":material/fact_check:"),
        st.Page("ui/page_export.py", url_path="export", title="3 · Export to Excel", icon=":material/table_view:"),
    ],
    "Model (researchers)": [
        st.Page("ui/page_library.py", url_path="training-data", title="Training data", icon=":material/folder_open:"),
        st.Page("ui/page_train.py", url_path="train", title="Train a model", icon=":material/model_training:"),
        st.Page("ui/page_benchmark.py", url_path="benchmark", title="Check vs hand-measured %", icon=":material/rule:"),
        st.Page("ui/page_models.py", url_path="models", title="Models", icon=":material/deployed_code:"),
        st.Page("ui/page_tuning.py", url_path="tune", title="Tune focus & specks", icon=":material/tune:"),
    ],
    "Help": [st.Page("ui/page_help.py", url_path="help", title="Help & settings", icon=":material/help:")],
}
page = st.navigation(pages)
common.sidebar()
page.run()
