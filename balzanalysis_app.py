import streamlit as st
import sys
import os

st.set_page_config(page_title="BalzAnalysis", layout="wide")

# --- Login Page ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.title("🔒 Login to BalzAnalysis")
    with st.form("login_form"):
        user_id = st.text_input("User ID")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if user_id == "bs0006" and password == "0006":
                st.session_state['logged_in'] = True
                st.success("Login successful! Please select a module from the sidebar.")
                st.rerun()
            else:
                st.error("Invalid ID or password.")
    st.stop()

# Custom CSS to make the sidebar less wide
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 180px;
        max-width: 180px;
        width: 180px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Remove the label from the sidebar radio
page = st.sidebar.radio(label="", options=["Options Analyzer", "Stock Analyzer", "Live Option Chain (NSE)"])

if page == "Options Analyzer":
    # Show the preferred header
    st.title("Balram Nifty50 Options Analyzer")
    # Add OptionChain/src to sys.path for local imports
    optionchain_src_path = os.path.join(os.path.dirname(__file__), "OptionChain", "src")
    sys.path.insert(0, optionchain_src_path)
    optionchain_app_path = os.path.join(optionchain_src_path, "app.py")
    with open(optionchain_app_path, encoding="utf-8") as f:
        code = f.read()
    exec(code, globals())
    sys.path.pop(0)

elif page == "Stock Analyzer":
    # Dynamically run StockAnalysis stock_analyzer_app.py code
    stock_app_path = os.path.join(os.path.dirname(__file__), "StockAnalysis", "stock_analyzer_app.py")
    with open(stock_app_path, encoding="utf-8") as f:
        code = f.read()
    exec(code, globals())

elif page == "Live Option Chain (NSE)":
    live_option_chain_path = os.path.join(os.path.dirname(__file__), "OptionChain", "src", "live_option_chain.py")
    with open(live_option_chain_path, encoding="utf-8") as f:
        code = f.read()
    exec(code, globals())
