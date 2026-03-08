import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import os

# Page title
st.title("📊 AI Data Analyst Agent")
st.write("Upload a CSV dataset and ask questions about your data.")

# OpenAI API key
api_key = st.text_input("Enter your OpenAI API Key", type="password")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None and api_key:

    # Read dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Columns")
    st.write(list(df.columns))

    # Create LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=api_key
    )

    # Create agent
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True
    )

    st.subheader("Ask a question about your data")

    # Question input box
    query = st.text_input("Type your question here")

    if query:
        with st.spinner("Analyzing data..."):
            response = agent.run(query)

        st.subheader("Answer")
        st.write(response)

    # Simple visualization section
    st.subheader("Quick Visualization")

    column = st.selectbox("Select column for visualization", df.columns)

    if column:
        st.bar_chart(df[column].value_counts().head(20))

else:
    st.info("Please upload a CSV file and enter your OpenAI API key.")
