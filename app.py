import streamlit as st
import pandas as pd
import numpy as np
import joblib
from prophet.plot import plot_plotly
import plotly.express as px

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Smart ML Analytics Dashboard",
    layout="wide"
)

# =========================
# TITLE
# =========================
st.title("Smart ML Analytics Dashboard")

# =========================
# SIDEBAR MENU
# =========================
menu = st.sidebar.selectbox(
    "Select Module",
    [
        "Home",
        "Shipment Prediction",
        "Customer Segmentation",
        "Demand Forecasting"
    ]
)

# =========================================================
# HOME
# =========================================================
if menu == "Home":

    st.header("Machine Learning Dashboard")

    st.write("""
    ### Modules Included
    - Supervised Learning
    - Unsupervised Learning
    - Time Series Forecasting
    """)

# =========================================================
# SHIPMENT PREDICTION
# =========================================================
elif menu == "Shipment Prediction":

    st.header("Shipment Delivery Prediction")

    # LOAD MODEL
    model = joblib.load("shipment_model.pkl")

    # USER INPUTS
    warehouse = st.selectbox(
        "Warehouse Block",
        [0, 1, 2, 3, 4]
    )

    shipment = st.selectbox(
        "Mode of Shipment",
        [0, 1, 2]
    )

    care_calls = st.slider(
        "Customer Care Calls",
        1,
        10,
        4
    )

    rating = st.slider(
        "Customer Rating",
        1,
        5,
        3
    )

    cost = st.number_input(
        "Cost of Product",
        min_value=50,
        max_value=500,
        value=200
    )

    prior = st.slider(
        "Prior Purchases",
        1,
        10,
        3
    )

    importance = st.selectbox(
        "Product Importance",
        [0, 1, 2]
    )

    gender = st.selectbox(
        "Gender",
        [0, 1]
    )

    discount = st.slider(
        "Discount Offered",
        0,
        100,
        10
    )

    weight = st.number_input(
        "Weight in gms",
        min_value=100,
        max_value=6000,
        value=2000
    )

    # CREATE INPUT ARRAY
    input_data = np.array([[
        warehouse,
        shipment,
        care_calls,
        rating,
        cost,
        prior,
        importance,
        gender,
        discount,
        weight
    ]])

    # PREDICTION BUTTON
    if st.button("Predict"):

        prediction = model.predict(input_data)

        if prediction[0] == 1:

            st.error(
                "Prediction Result: Shipment may be delayed based on shipment details."
            )

            st.write("""
            ### Possible Reasons
            - High shipment weight
            - Delivery mode
            - Product handling conditions
            - Customer history
            """)

        else:

            st.success(
                "Prediction Result: Shipment is expected to arrive on time."
            )

# =========================================================
# CUSTOMER SEGMENTATION
# =========================================================
elif menu == "Customer Segmentation":

    st.header("Customer Segmentation")

    # LOAD DATASET
    mall_df = pd.read_csv("Mall_Customers.csv")

    # FEATURES
    X = mall_df[
        ['Annual Income (k$)', 'Spending Score (1-100)']
    ]

    # LOAD MODEL
    kmeans = joblib.load("kmeans_model.pkl")

    # PREDICT CLUSTERS
    mall_df['Cluster'] = kmeans.predict(X)

    # SCATTER PLOT
    fig = px.scatter(
        mall_df,
        x='Annual Income (k$)',
        y='Spending Score (1-100)',
        color='Cluster',
        title='Customer Segments'
    )

    st.plotly_chart(fig)

    # DISPLAY DATA
    st.write(mall_df)

# =========================================================
# DEMAND FORECASTING
# =========================================================
elif menu == "Demand Forecasting":

    st.header("Demand Forecasting")

    # LOAD MODEL
    model = joblib.load("demand_model.pkl")

    # SELECT DAYS
    days = st.slider(
        "Forecast Days",
        7,
        365,
        90
    )

    # CREATE FUTURE DATAFRAME
    future = model.make_future_dataframe(
        periods=days
    )

    # PREDICT FORECAST
    forecast = model.predict(future)

    # PLOT FORECAST
    fig = plot_plotly(
        model,
        forecast
    )

    st.plotly_chart(fig)

    # DISPLAY FORECAST TABLE
    st.write(
        forecast[['ds', 'yhat']].tail(days)
    )
