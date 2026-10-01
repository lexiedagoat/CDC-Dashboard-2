"""
app.py
Main Streamlit execution script bringing together headers, filters, KPIs, and tabs.
"""

import streamlit as st
from components.kpi_cards import render_kpi_cards
from components.sidebar import render_sidebar
from components.tabs import (
    render_about_tab,
    render_data_table_tab,
    render_geographic_tab,
    render_monthly_sex_tab,
    render_overview_tab,
)
from data_loader import load_data

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="CDC 2025 Provisional Natality Dashboard",
    page_icon="📊",
    layout="wide",
)

# Render Header Component
st.title("CDC Provisional Natality Explorer (2025)")
st.markdown(
    """
*An interactive analytics portal for examining geographic, temporal, and sex-based birth counts across the United States.*
"""
)

# Notice Callout Requirements
st.warning(
    "**NOTICE**: The figures displayed in this dashboard represent **provisional raw birth counts**, not finalized figures or population-adjusted birth rates. Data source: **CDC WONDER**."
)

# Load Data
df = load_data()

# Render Sidebar and retrieve filtered subset
filtered_df = render_sidebar(df)

# Render Top KPI Summary Cards
render_kpi_cards(filtered_df)

st.markdown("---")

# Render Main Analytics Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Overview",
        "Geographic Analysis",
        "Monthly & Sex Analysis",
        "Data Table & Download",
        "About the Data",
    ]
)

with tab1:
    render_overview_tab(filtered_df)

with tab2:
    render_geographic_tab(filtered_df)

with tab3:
    render_monthly_sex_tab(filtered_df)

with tab4:
    render_data_table_tab(filtered_df)

with tab5:
    render_about_tab()
