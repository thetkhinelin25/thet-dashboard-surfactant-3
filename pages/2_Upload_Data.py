import streamlit as st
import pandas as pd

st.set_page_config(page_title="Upload Data", layout="wide")
st.title("Page 2: Upload Data in excel format")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, sheet_name="Data_Summary_PST")
    st.session_state["shared_df"] = df

if "shared_df" in st.session_state:
    st.markdown("✅ File successfully uploaded")
else:
    st.warning("⚠️ Excel file is not uploaded yet!")

