"""Entry point for the multi-page app (Streamlit Cloud runs this file).

st.navigation lets us set each page's sidebar title and icon explicitly.
Without it, Streamlit names the main page after its file ("app") and the
others after their filenames in pages/.
"""
import streamlit as st

pg = st.navigation(
    [
        st.Page("flowers.py", title="Fredonia Flowers", icon="🌷", default=True),
        st.Page("pages/1_ITS_Assistant.py", title="ITS Assistant", icon="💻"),
        st.Page("pages/2_Staff_Scheduler.py", title="Staff Scheduler", icon="🩺"),
        st.Page("pages/3_Invoice_Tracker.py", title="Invoice Tracker", icon="🧾"),
    ]
)
pg.run()
