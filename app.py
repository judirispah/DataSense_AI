import pandas as pd
import streamlit as st
import json
import os 
from dotenv import load_dotenv
import numpy as np
from langchain_groq import ChatGroq



##load groq api

load_dotenv()
groq_api_key=os.getenv('GROQ_API_KEY')   

st.set_page_config(layout="wide")


st.title("🤖 DataSense AI: Turning Data into Intelligence")


# File uploader
uploaded_file = st.file_uploader("Upload a CSV, Excel, or JSON file", type=["csv", "xlsx", "xls", "json"])


if uploaded_file is not None:
        filename = uploaded_file.name
        file_type=filename.split('.')[-1].lower()
        print(file_type)
        print(type(uploaded_file))
        try:
            if file_type == 'csv':
                df = pd.read_csv(uploaded_file,na_values=['NA', 'null', '-' ,'NIL','None'])
        #------------------------------------------------------------            
            elif file_type in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file,na_values=['NA', 'null', '-' ,'NIL','None'])
        #------------------------------------------------------------            

            elif file_type == 'json':
                
               try:
                    #case 1: If it's a list of dict 
                    df = pd.read_json(uploaded_file, lines=False)
               except ValueError:
                    #case 1: If it's a newline-delimited dict or single dict   
                    uploaded_file.seek(0)#reset it with seek(0) if you want to read it again!
                    df = pd.read_json(uploaded_file, lines=True)

                
                

                
                # Case 2: If it's a list of dicts (JSON array), convert directly
                #elif isinstance(data,list):
                     #df=pd.DataFrame(data)
                

        #------------------------------------------------------------  
                          
            else:
                st.error("Unsupported file type")
            st.success(f"{file_type.upper()} file successfully loaded!")  
            st.dataframe(df.head())
        #------------------------------------------------------------  

            shape=df.shape
            summary_stats = df.describe()
            null_counts = df.isna().sum().div(df.shape[0]).mul(100).to_frame(name='MissingPercentage').sort_values(by='MissingPercentage',ascending=False)

            #correlation_matrix = df.corr().to_string()
            numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O']
            categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']
            

            prompt = f"""
            You are a data scientist.Perform Exploratory Data Analysis (EDA) on the uploaded data.Make a standard format 
            for high visibility of analysis

                Analyze this dataset in this format:

                1.shape:{shape}
                2.Missing Values:{null_counts}
                3.numeric_features:{numeric_features}
                4.categorical_features:{categorical_features}
                5.summary_statistics (in table):{summary_stats}
                

                Give a concise , missing issues, and conclusion IN tablular format with reports in beautful   words.
                """

            llm = ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.3-70b-versatile")
            response = llm.invoke(prompt)
            st.title('Exploratory Data Analysis (EDA):')

            st.info(response.content)
            st.balloons()

            

            option = st.selectbox(
                    "select the target column",
                    [column for column in df.columns],index=None
                                    )

            st.write("You selected:", option)
             

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        except Exception as e:
            st.error(f"Error loading file: {e}")          
        




















else:
    st.info("Please upload a structured data file to begin.") 




