# main.py (aka Welcome_Page.py)
import streamlit as st

st.set_page_config(page_title="Welcome page", layout="wide")

st.title("Welcome to the Surfactant App")
st.markdown("""
Use the sidebar to navigate between different pages:
- **User Guide**
- **Upload Data**
- **UMAP Exploration**
- **Performances Prediction**
- **Formulation Suggestion**
""")
