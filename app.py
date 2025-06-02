import pandas as pd
import streamlit as st
import json
import numpy as np

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

        except Exception as e:
            st.error(f"Error loading file: {e}")          
        
else:
    st.info("Please upload a structured data file to begin.") 




