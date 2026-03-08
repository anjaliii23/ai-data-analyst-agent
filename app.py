import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="AI Data Analyst Agent", page_icon="📊")

st.title("📊 AI Data Analyst Agent")
st.write("Upload a CSV dataset and ask questions about your data.")

# Get API key from Streamlit secrets
api_key = "YOUR_NEW_OPENAI_KEY"
# Upload dataset
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Columns")
    st.write(df.columns.tolist())

    # User question
    question = st.text_input("Ask a question about your data")

    if question:

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=api_key
        )

        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            allow_dangerous_code=True
        )

        with st.spinner("Analyzing data..."):

            response = agent.run(question)

        st.subheader("AI Insight")
        st.write(response)

    # Visualization section
    st.subheader("Quick Visualization")

    column = st.selectbox("Select column for visualization", df.columns)

    if pd.api.types.is_numeric_dtype(df[column]):
        st.bar_chart(df[column])
    else:
        st.write("Selected column is not numeric. Please choose a numeric column.")