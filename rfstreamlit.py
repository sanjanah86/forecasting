import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import plotly.graph_objects as go
import uuid
from functions import forecast_date_picker,run_forecast,resample_forecast,get_filters,reset_filters,plot_lpmpd_dual_line
from data_processing import normalize_columns, preprocess_data, aggregate_data, save_processed_data 
from style_config import apply_styles,add_download_button
 
apply_styles()
 
@st.cache_resource
def load_models():
    machine_model = joblib.load("rf_machine_model.pkl")
    substrate_model = joblib.load("rf_substrate_model.pkl")
    return machine_model, substrate_model
 
machine_model, substrate_model = load_models()

uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
 
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['batch_date'])
    df = normalize_columns(df)  # Your preprocessing functions
    df = preprocess_data(df)
    df['batch_date'] = pd.to_datetime(df['batch_date'])
    df = df.sort_values('batch_date')
 
    st.sidebar.header("Preferences")
    freq_option = st.sidebar.radio("Aggregation Frequency", ["Daily", "Weekly", "Monthly", "Quarterly"])
    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q"}
    freq = freq_map[freq_option]
  
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "machine_group"
    st.markdown("### Select Grouping to calculate LPMPD")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Machine Group", type="primary",width="stretch"):
            st.session_state.active_tab = "machine_group"
    with col2:
        if st.button("Substrate",type="primary",width="stretch"):
            st.session_state.active_tab = "substrate_code"
    with col3:
        if st.button("Machine & Substrate",type="primary",width="stretch"):
            st.session_state.active_tab = "machine_substrate"

    active_tab = st.session_state.active_tab

    if active_tab == "machine_group":
        group_cols = "machine_group"
        agg_df = aggregate_data(df, method="machine_group", freq=freq)
        model = machine_model
 
    elif active_tab == "substrate_code":
        group_cols = "substrate_code"
        agg_df = aggregate_data(df, method="substrate_code", freq=freq)
        model = substrate_model
 
    else:
        group_cols = ["machine_group", "substrate_code"]
        agg_df = aggregate_data(df, method="machine_substrate", freq=freq)
        model = machine_model  
 
    st.dataframe(agg_df)

    st.subheader("Forecast Results")
    
    forecast_start, forecast_end = forecast_date_picker(df)

    if 'selected_group' not in st.session_state:
        st.session_state.selected_group = "All"
    if 'selected_code' not in st.session_state:
        st.session_state.selected_code = "All"

    filters = get_filters(group_cols)
    groups = df['machine_group'].dropna().unique().tolist() if 'machine_group' in filters else []
    codes = df['substrate_code'].dropna().unique().tolist() if 'substrate_code' in filters else []
 
    selected_group = st.selectbox("Filter by Machine Group", ["All"] + groups, index=0, key='selected_group') if 'machine_group' in filters else "All"
    selected_code = st.selectbox("Filter by Substrate Code", ["All"] + codes, index=0, key='selected_code') if 'substrate_code' in filters else "All"

    st.button("Reset Filters", on_click=reset_filters)
 
    
    forecast_df = run_forecast(agg_df, forecast_start, forecast_end, model, group_cols, freq)
    forecast_df = resample_forecast(forecast_df,freq_option,group_cols)

    if forecast_df is not None:
        if selected_group != "All" and "machine_group" in forecast_df.columns:
            forecast_df = forecast_df[forecast_df["machine_group"] == selected_group]
        if selected_code != "All" and "substrate_code" in forecast_df.columns:
             forecast_df = forecast_df[forecast_df["substrate_code"] == selected_code]
 
        st.write("### Forecasted Data")
        cols_to_show = ['batch_date'] + (group_cols if isinstance(group_cols, list) else [group_cols]) + ['forecasted_lpmpd']
        st.dataframe(forecast_df[cols_to_show])
        forecast_df=forecast_df[cols_to_show]
    else:
        st.write("No forecast data available to display.")
          
    add_download_button(forecast_df, active_tab)
    if forecast_df is not None:


        plot_lpmpd_dual_line(agg_df, forecast_df, active_tab=active_tab)
