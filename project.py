import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from streamlit_option_menu import option_menu
from sklearn.linear_model import  LinearRegression
import plotly.express as px

df=pd.read_csv("custom_dataset.csv")
#customerdata
data=pd.read_csv("Customer Data.csv")
data1=pd.read_csv("Mall_Customers.csv")


# Sidebar option menu
with st.sidebar:
    selected = option_menu(
        menu_title="Go to",
        options= ["🏠 Project Overview", "📊 Segmentation Dashboard", "🔮 Prediction Dashboard", "📈 Analytics"],
        icons=["🏠 ","📊", "🔮", "📈 "],
        menu_icon="cast",
        default_index=0,

    )
if selected=="🏠 Project Overview":
    st.markdown("""
        <h1 style='text-align: left; color: #0e76a8;'>
            📊 Project Overview
        </h1>
    """, unsafe_allow_html=True)
    # Apply custom styles


    # Use HTML to apply CSS classes

    st.markdown("<h2 class='section-header'>Project Summary</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p class='section-text'>Intelligent Customer Segmentation and Predictive Insights for B2C Growth</p>",
        unsafe_allow_html=True
    )

    st.markdown("<h2 class='section-header'>Objective</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p class='section-text'>The objective of this project is to analyze customers, segment them into distinct groups, and predict their future actions to enable personalized marketing strategies and improved customer engagement.</p>",
        unsafe_allow_html=True
    )

    st.markdown("<h2 class='section-header'>Overview</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p class='section-text'>In modern B2C businesses, customer understanding is critical for growth. This project leverages machine learning to segment customers based on demographics and purchasing behavior. A future extension will include predictive analytics to forecast customer purchases, churn, and recommend tailored offers.</p>",
        unsafe_allow_html=True,
    )
if selected=="📊 Segmentation Dashboard":
    st.markdown("""
        <style>
            h1, h2 {
                text-align: left;
                color: #0e76a8;
                margin-bottom: 10px;
            }

            .step-header {
                font-size: 22px;
                font-weight: 600;
                color: #444;
                margin-top: 30px;
                margin-bottom: 10px;
            }

            .bi {
                margin-right: 8px;
            }

            /* Optional: style slider label */
            .stSlider > label {
                font-weight: 500;
                color: #333;
            }
        </style>

        <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css" rel="stylesheet">
    """, unsafe_allow_html=True)

    # Title
    st.markdown("<h1><i class='bi bi-graph-up'></i> Customer Segmentation Dashboard</h1>", unsafe_allow_html=True)

    # Step 1 - Load Data
    st.markdown("<h2 class='step-header'><i class='bi bi-upload'></i> Step 1: Load Data</h2>", unsafe_allow_html=True)
    st.dataframe(df)

    # Step 2 - Clustering
    st.markdown("<h2 class='step-header'><i class='bi bi-diagram-3'></i> Step 2: Perform Clustering</h2>",
                unsafe_allow_html=True)

    # Cluster Slider
    degree = st.slider("Please Enter the number of Clusters", 1, 5, 2)

    # Apply clustering
    x = df[['Age', 'Income', 'Score']]
    kmeans = KMeans(n_clusters=degree, n_init=10)
    df['cluster'] = kmeans.fit_predict(x)
    st.dataframe(df)

    # Step 3 - Visualisation
    st.markdown("<h2 class='step-header'><i class='bi bi-scatter-chart'></i> Step 3: Cluster Visualisation</h2>",
                unsafe_allow_html=True)

    # Plot clusters
    fig, ax = plt.subplots()
    scatter = ax.scatter(df['Age'], df['Income'], c=df['cluster'], cmap='viridis')
    ax.set_xlabel('Age')
    ax.set_ylabel('Income')
    ax.set_title('Clusters based on Age and Income')
    st.pyplot(fig)

if selected== "🔮 Prediction Dashboard":
    reg = linear_model.LinearRegression()
    reg.fit(df[["Age","Income"]],df["Score"])
    with st.form(key="form1"):
        age=st.text_input("Enter a Age")
        income=st.text_input("Enter a Income")
        submit=st.form_submit_button("CLICK")
        if submit:
            Age=float(age)
            Income=float(income)
            pre=reg.predict([[Age,Income]])
            st.write(pre)

if selected=="📈 Analytics":
    st.markdown("""
        <style>
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                text-align: left;
            }
            h1, h2, h3, h4 {
                color: #2E86C1;
            }
            .stTextInput, .stNumberInput {
                border-radius: 8px;
                padding: 5px;
            }
            .stButton > button {
                background-color: #3498DB;
                color: white;
                border-radius: 8px;
                padding: 10px 24px;
            }
            .stButton > button:hover {
                background-color: #2E86C1;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("## 📈 Welcome To Customer Analytics")
    st.markdown("### 🚀 Customer Mall Analysis")
    st.markdown("### 📅 Annual Income (k$) vs Spending Score (1-100)")

    kmeans = KMeans(n_clusters=3, n_init=10)
    data1["cluster"] = kmeans.fit_predict(data1[["Annual Income (k$)", "Spending Score (1-100)", "Age"]])

    fig = px.scatter(data1, x="Annual Income (k$)", y="Spending Score (1-100)", color="cluster",
                     title="Age vs Spending Score (1-100)", color_continuous_scale="viridis")
    fig.update_layout(xaxis_title="Age (Years)", yaxis_title="Spending Score")
    st.plotly_chart(fig)

    with st.form(key="form2"):
        st.markdown("### 📉 Income-wise Spending Cluster")
        Age = st.text_input("Enter Age")
        spendigscore = st.number_input("Enter Spending Score")
        anualincome = st.number_input("Enter Annual Income")
        submit = st.form_submit_button(label="🔎 Predict Cluster")

        if submit:
            age = float(Age)
            spendigscore = float(spendigscore)
            anualincome = float(anualincome)
            pre = kmeans.predict([[anualincome, spendigscore, age]])
            st.success(f"Predicted Cluster: {pre[0]}")

    st.markdown("### 📊 Gender vs Spending Score")
    count = data1.groupby("Gender")["Spending Score (1-100)"].mean()
    st.write(count)

    fig = px.bar(count, x=["Male", "Female"], y="Spending Score (1-100)",
                 color_discrete_map={"Male": "pink", "Female": "blue"},color=["Male","Female"])
    fig.update_layout(xaxis_title="Gender", yaxis_title="Spending Score (1-100)", width=1000, height=500)
    st.plotly_chart(fig)

    st.markdown("### 🌐 Age Group vs Spending Score")
    bins = [0, 20, 30, 40, 50, 60, 70]
    group = ["0-20", "20-30", "30-40", "40-50", "50-60", "60-70"]
    data1["Agegroup"] = pd.cut(data1["Age"], bins=bins, labels=group, ordered=False)
    groupage = data1.groupby("Agegroup")["Spending Score (1-100)"].mean()

    fig = px.bar(groupage, x=group, y="Spending Score (1-100)",
                 color=group, barmode='group',
                 color_discrete_map={"0-20": "red", "20-30": "yellow", "30-40": "red",
                                     "40-50": "pink", "50-60": "green", "60-70": "black"})
    fig.update_layout(xaxis_title="Age Groups", yaxis_title="Spending Score (1-100)", width=1000, height=500)
    st.plotly_chart(fig)

    st.markdown("### 🔹 Scatter: Income vs Score")
    fig = px.scatter(data1, x="Annual Income (k$)", y="Spending Score (1-100)")
    fig.update_layout(xaxis_title="Annual Income (k$)", yaxis_title="Spending Score (1-100)",  bargap=0.1)
    st.plotly_chart(fig,use_container_width=True)

    st.markdown("### 📊 Histogram: Age")
    fig = px.histogram(data1, x="Age")
    fig.update_layout(
        xaxis_title='Age Range',
        yaxis_title='Frequency',
        bargap=0.1,

        template='plotly_white'
    )
    fig.update_traces(ybins=dict(start=0, end=20, size=5))
    st.plotly_chart(fig)

