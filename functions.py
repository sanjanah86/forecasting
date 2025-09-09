import pandas as pd
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import plotly.express as px
import streamlit as st
#-----------------------------------------------------

def forecast_date_picker(df):
    last_date = df['batch_date'].max().date()
    start_date = last_date + timedelta(days=1)
    end_date = last_date + relativedelta(months=3)
    selected_start = st.date_input("Forecast Start Date", min_value=start_date, max_value=end_date, value=start_date)
    selected_end = st.date_input("Forecast End Date", min_value=selected_start, max_value=end_date, value=end_date)
    return pd.to_datetime(selected_start), pd.to_datetime(selected_end)

#-----------------------------------------------------

def run_forecast(df, forecast_start, forecast_end, model, group_cols=None, freq='D'):
    df_copy = df.copy()
    #actual_data = df[['batch_date', 'lpmpd']].copy()
    #X = df.drop(['batch_date', 'lpmpd'], axis=1)  
    #X_scaled = scaler.transform(X)  
    #predictions = model.predict(X_scaled).flatten()
    #pred_df = actual_data.copy()
    #pred_df['predicted'] = predictions
    future_dates = pd.date_range(start=forecast_start, end=forecast_end, freq=freq)
    if group_cols is None:
        future_df = pd.DataFrame({"batch_date": future_dates})
    elif isinstance(group_cols, list):
        unique_vals = df_copy[group_cols].drop_duplicates()
        future_df = pd.DataFrame(
            [{**dict(zip(group_cols, row)), "batch_date": d} for d in future_dates for row in unique_vals.values]
        )
    else:
        unique_vals = df_copy[group_cols].unique()
        future_df = pd.DataFrame(
            [{"batch_date": d, group_cols: g} for d in future_dates for g in unique_vals]
        )
 
    future_df['month'] = future_df['batch_date'].dt.month
    future_df['day'] = future_df['batch_date'].dt.day
    future_df['dayofweek'] = future_df['batch_date'].dt.dayofweek
 
    if group_cols:
        group_means = df_copy.groupby(group_cols)['batch_count'].mean().reset_index()
        future_df = future_df.merge(group_means, on=group_cols, how='left')
    else:
        future_df['batch_count'] = df_copy['batch_count'].mean()
 
    feature_cols = ['month', 'day', 'dayofweek', 'batch_count']
    future_df['forecasted_lpmpd'] = model.predict(future_df[feature_cols]).round(2)
    return future_df
    st.write(future_df)

#-----------------------------------------------------

def get_filters(group_cols):
    filters = []
    if isinstance(group_cols, list):
        filters.extend(['machine_group', 'substrate_code'])
    else:
        if group_cols == 'machine_group':
            filters.append('machine_group')
        elif group_cols == 'substrate_code':
            filters.append('substrate_code')
    return filters

#-----------------------------------------------------

def reset_filters():
    st.session_state.selected_group = "All"
    st.session_state.selected_code = "All"

#-----------------------------------------------------

def resample_forecast(forecast_df, freq_option, group_cols):
    

    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q"}

    df = forecast_df.copy()
    df['batch_date'] = pd.to_datetime(df['batch_date'])

    # Ensure group_cols is a list
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    if freq_option == "Daily":
        df['batch_date'] = df['batch_date'].dt.strftime('%Y-%m-%d')
        df['forecasted_lpmpd'] = df['forecasted_lpmpd'].round(2)

    else:
        df = (df.groupby(group_cols).resample(freq_map[freq_option], on='batch_date')['forecasted_lpmpd'].mean().round(2).reset_index())
        if freq_option == "Monthly":
            df['batch_date'] = df['batch_date'].dt.strftime('%B %Y')

        elif freq_option == "Quarterly":
            df['Quarter'] = df['batch_date'].dt.quarter
            df['Year'] = df['batch_date'].dt.year
            df['batch_date'] = "Q" + df['Quarter'].astype(str) + " " + df['Year'].astype(str)
            df.drop(columns=['Quarter', 'Year'], inplace=True)

        elif freq_option == "Weekly":
            df['Year'] = df['batch_date'].dt.year
            df['week_num'] = df['batch_date'].dt.isocalendar().week
            df['batch_date'] = "W" + df['week_num'].astype(str).str.zfill(2) + " " + df['Year'].astype(str)
            df.drop(columns=['week_num', 'Year'], inplace=True)

    return df



#-----------------------------------------------------


def plot_lpmpd_dual_line(agg_df, forecast_df, active_tab="machine_group", selected_group="All", selected_code="All"):
    
    hist_plot = agg_df.copy()
    forecast_plot = forecast_df.copy()

    # Apply filters
    if selected_group != "All" and "machine_group" in hist_plot.columns:
        hist_plot = hist_plot[hist_plot["machine_group"] == selected_group]
        forecast_plot = forecast_plot[forecast_plot["machine_group"] == selected_group]

    if selected_code != "All" and "substrate_code" in hist_plot.columns:
        hist_plot = hist_plot[hist_plot["substrate_code"] == selected_code]
        forecast_plot = forecast_plot[forecast_plot["substrate_code"] == selected_code]

    # Keep the grouping columns before selecting other columns
    if active_tab == "machine_group":
        group_cols = ['machine_group']
    elif active_tab == "substrate_code":
        group_cols = ['substrate_code']
    elif active_tab == "machine_substrate":
        group_cols = ['machine_group', 'substrate_code']
    else:
        group_cols = []

    # Prepare historical data with grouping columns included
    hist_plot = hist_plot[['batch_date', 'lpmpd'] + group_cols].copy()
    hist_plot.rename(columns={'lpmpd':'Value'}, inplace=True)
    hist_plot['Type'] = 'Historical'

    # Prepare forecast data with grouping columns included
    forecast_plot = forecast_plot[['batch_date', 'forecasted_lpmpd'] + group_cols].copy()
    forecast_plot.rename(columns={'forecasted_lpmpd':'Value'}, inplace=True)
    forecast_plot['Type'] = 'Forecasted'

    # Assign Group column based on active_tab
    if active_tab == "machine_group":
        hist_plot['Group'] = hist_plot['machine_group']
        forecast_plot['Group'] = forecast_plot['machine_group']
    elif active_tab == "substrate_code":
        hist_plot['Group'] = hist_plot['substrate_code']
        forecast_plot['Group'] = forecast_plot['substrate_code']
    elif active_tab == "machine_substrate":
        hist_plot['Group'] = hist_plot['machine_group'] + " | " + hist_plot['substrate_code']
        forecast_plot['Group'] = forecast_plot['machine_group'] + " | " + forecast_plot['substrate_code']
    else:
        hist_plot['Group'] = 'All'
        forecast_plot['Group'] = 'All'

    # Combine datasets
    plot_df = pd.concat([hist_plot, forecast_plot], ignore_index=True)

    # Create the plot
    fig = px.line(
        plot_df,
        x='batch_date',
        y='Value',
        color='Type',
        line_dash='Type',
        facet_col='Group' if active_tab == "machine_substrate" else None,
        labels={'Value':'LPMPD', 'batch_date':'Date', 'Type':'Data Type'},
        title='Historical vs Forecasted LPMPD'
    )

    st.plotly_chart(fig, use_container_width=True)
