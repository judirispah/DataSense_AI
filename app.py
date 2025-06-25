import pandas as pd
import streamlit as st
import json
import os 
from dotenv import load_dotenv
import numpy as np
from langchain_groq import ChatGroq
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway





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
            summary_stats = df.describe().to_dict()
            null_counts = df.isna().sum().div(df.shape[0]).mul(100).to_frame(name='MissingPercentage').sort_values(by='MissingPercentage',ascending=False).to_dict()

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
                5.summary statistics:{summary_stats}
                

                Give a concise , missing issues, and conclusion IN tablular format with reports in beautful   words with no recommdations.
                """

            llm = ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.3-70b-versatile")
            response1 = llm.invoke(prompt)
            st.title('Exploratory Data Analysis (EDA):')

            st.info(response1.content)
            st.balloons()



            

        #------------------------------------------------------------------------------------
            st.title('Univariate Analysis:')
            column = st.selectbox("Select column for univariate analysis", df.columns,index=None)
            if column is not None and column in numeric_features:
                 plot_option = st.selectbox("Select plot", ["Histogram", "Boxplot", "Density Plot"])
                 #plt.figure(figsize=(15, 8))
                 fig, ax = plt.subplots()
                 if plot_option == "Histogram":
                      sns.histplot(df[column], kde=True, ax=ax)
                 elif plot_option == "Boxplot":
                      sns.boxplot(y=df[column], ax=ax)
                 elif plot_option == "Density Plot":
                      sns.kdeplot(df[column], ax=ax)
                 st.pyplot(fig) 
                 desc = df[column].describe()
                 value_counts = df[column].value_counts().to_dict()
                 iqr = desc["75%"] - desc["25%"]
                 outliers = df[(df[column] < desc["25%"] - 1.5 * iqr) | (df[column] > desc["75%"] + 1.5 * iqr)]
                 print(outliers)
                 outlier_count = len(outliers)
                 skew_value=df[column].skew()
                 univarient_num_prompt=f"""
                                    Analyze this univariate column:
                                    - Summary stats: {desc.to_dict()}
                                    -skew value: {skew_value}
                                    
                                    -outliers: {outliers[column]}(using IQR method){iqr}
                                    - Outliers detected: {outlier_count}

                                    Please provide:
                                        - Insights on distribution (normal/skewed)
                                        - Outlier explanation
                                    all in undersatbale and simple words and table formats without long paragraphs and recoommdations with heading"""  
                 response2=llm.invoke(univarient_num_prompt)  
                 st.info(response2.content) 

            if column is not None and column in categorical_features:
                 plot_option = st.selectbox("Select plot", ["Bar Chart", "Pie Chart"],)
                 #plt.figure(figsize=(8, 8))
                 fig, ax = plt.subplots()
                 if plot_option == "Bar Chart":
                      sns.countplot(data=df, x=column, ax=ax)
                 elif plot_option == "Pie Chart":
                      
                      df[column].value_counts().plot.pie(y=df[column],figsize=(15,16),autopct='%1.1f')
                      #ax.axis('equal')  # Ensures the pie is a circle
                 st.pyplot(fig)
                 counts = df[column].value_counts().to_dict()
                 univarient_cat_prompt = f"Here is the category distribution as percentages: {counts}. Give me insights about this distribution with heading."

                 response3=llm.invoke(univarient_cat_prompt)  
                 st.info(response3.content) 

        #------------------------------------------------------------------------------------
            st.title('Bivariate Analysis:')
            col1, col2 = st.columns(2)

            with col1:
               column1 = st.selectbox("Select first column for Bivariate analysis", df.columns, index=None)

            with col2:
               column2 = st.selectbox("Select second column for Bivariate analysis", df.columns, index=None)

            #num and num
            if (column1 in numeric_features and not None) and (column2 in numeric_features and not None):
                 plot_option = st.selectbox("Select plot", ["Scatter Plot", "Line Plot","Regression Plot"],)
                 fig, ax = plt.subplots()

                 if plot_option == "Scatter Plot":
                    sns.scatterplot(data=df, x=column1, y=column2, ax=ax)

                 elif plot_option == "Line Plot":
                    df_sorted = df.sort_values(by=column1)
                    sns.lineplot(data=df_sorted, x=column1, y=column2, ax=ax)

                 elif plot_option == "Regression Plot":
                    sns.regplot(data=df, x=column1, y=column2, ax=ax, scatter_kws={"alpha": 0.5})

                 ax.set_title(f"{plot_option}: {column1} by {column2}")   

                 st.pyplot(fig)  
                 corr_value = df[[column1, column2]].corr().iloc[0,1]
                 print(corr_value) 
                 num_vs_num_prompt = f"""
                         You are a data analyst. Given a dataset with the following numerical features: {column1},{column2} 
                         perform a correlation analysis for pairs of numerical columns.

                         For pair:
                         1. the Pearson correlation coefficient (r) :{corr_value}.
                         2. Classify the relationship as:
                              - Strong Positive (r > 0.7)
                              - Moderate Positive (0.4 < r ≤ 0.7)
                              - Weak/No Correlation (−0.4 ≤ r ≤ 0.4)
                              - Moderate Negative (−0.7 ≤ r < −0.4)
                              - Strong Negative (r < −0.7)
                         3. Briefly explain the correlation and any possible real-world implications with a heading Correlation Analysis for numcerical com.
                         4. Highlight if the correlation suggests multicollinearity or a valuable predictive relationship breifly and shortly no long paragraph .
                         """
                 response4=llm.invoke(num_vs_num_prompt) 
                 st.info(response4.content) 

                 
            
            #num and cat
            if ((column1 in numeric_features  and not None) and (column2 in categorical_features and not None)) or ((column2 in numeric_features  and not None) and (column1 in categorical_features and not None)):
                 plot_option = st.selectbox("Select plot", ["Count Plot","Box Plot","Violin Plot","Strip Plot","Swarm Plot"],)
                 fig, ax = plt.subplots()

                # Decide orientation
                 if column1 in categorical_features:
                    cat_col, num_col = column1, column2
                 else:
                    cat_col, num_col = column2, column1

                 if plot_option == "Box Plot":
                    sns.boxplot(x=cat_col, y=num_col, data=df, ax=ax)
        
                 elif plot_option == "Violin Plot":
                    sns.violinplot(x=cat_col, y=num_col, data=df, ax=ax)
                 elif plot_option == "Strip Plot":
                    sns.stripplot(x=cat_col, y=num_col, data=df, ax=ax, jitter=True)
                 elif plot_option == "Swarm Plot":
                    sns.swarmplot(x=cat_col, y=num_col, data=df, ax=ax)
                 elif plot_option == "Count Plot":
                    sns.countplot(x=num_col,hue=cat_col, data=df, ec= "black",palette="Accent")
                        

                 ax.set_title(f"{plot_option}: {num_col} by {cat_col}")
                 st.pyplot(fig)
                 count=df.groupby(cat_col)[num_col].count().to_dict()
                 mean=df.groupby(cat_col)[num_col].mean().to_dict()
                 median=df.groupby(cat_col)[num_col].median().to_dict()

                 groups = [group[num_col].dropna().values for name, group in df.groupby(cat_col)]
                 # Run the ANOVA test
                 f_stat, p_val = f_oneway(*groups)

                 cat_vs_num_prompt=f"""
                     You are a data analyst. Given a dataset with the following categorical and numerical feature : {cat_col},{num_col} 
                         perform a correlation analysis for pairs of categorical and numerical  columns.
                         for pair:
                         1.count : {count}
                         2.mean : {mean}
                         3.median:{median}
                         4.Annova test(ANOVA helps you find out which numerical features are most useful in distinguishing between categories (classes)
                         mainly used for independnet vs depentent feature for feature imporantce to target variable):
                           - f_stat:{f_stat}
                           - p_value:{p_val}


                           Interpretation(provide conclusion in tabular format) :
                                     Null Hypothesis (H₀):
                                             All group means are equal. No significant difference between classes.
                                             The feature may not help in class separation.
                                             P-Value is greater than or equal to 0.05
                                             Fail to reject the null hypothesis (H₀)
                                             Consider removing or deprioritizing the feature.
                                    Alternative Hypothesis (H₁):
                                             At least one group mean is significantly different.
                                             This numerical feature is statistically important for distinguishing the classes.
                                             Use this feature in your classification model.
                                             P-Value is less than 0.05
                                             Reject the null hypothesis (H₀)
                                             Consider keeping or prioritizing the feature.
                                             

                        Breifly explain these analysis in readable and table format for easy analysis and headings."""
                 
                 response5=llm.invoke(cat_vs_num_prompt) 
                 st.info(response5.content) 

            # cat and cat
            if (column1 in categorical_features and not None) and (column2 in categorical_features and not None):
                 plot_option = st.selectbox("Select plot", ["Countplot", "Heatmap","Stacked Bar"],)
                 fig, ax = plt.subplots()
                 cat1,cat2=column1,column2

                 if plot_option == "Countplot":
                    sns.countplot(x=cat1, hue=cat2, data=df, ax=ax)
        
                 elif plot_option == "Heatmap":
                    ct = pd.crosstab(df[cat1], df[cat2])
                    sns.heatmap(ct, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
        
                 elif plot_option== "Stacked Bar":
                    ct = pd.crosstab(df[cat1], df[cat2], normalize='index')
                    ct.plot(kind='bar', stacked=True, ax=ax)

                 ax.set_title(f"{plot_option}: {cat1} by {cat2}")

                 st.pyplot(fig)
                 
                 ct_norm = (pd.crosstab(df[cat1], df[cat2], normalize='index') * 100).to_dict()
                 chi2, p, dof, expected = chi2_contingency(pd.crosstab(df[cat1], df[cat2]))

                 print(ct_norm)

                 cat_vs_cat_prompt=f"""You are a data analyst. Given a dataset with the following categorical features: {cat1},{cat2} 
                         perform a correlation analysis for pairs of categorical columns.
                         for pair:
                      
                         1..Normalized (row %) Table::{ct_norm}
                         2..Chi-Square Test(mainly used for independnet vs depentent feature for feature imporantce to target variable):
                                    Chi2 Value: {chi2:.2f}"
                                    Degrees of Freedom: {dof}"
                                    P-Value: {p:.4f}"
                                   
                                    Interpretation(in tabular format):
                                     Null Hypothesis (H₀):
                                             There is no association between the two categorical variables.
                                             In other words, the categrical variables are independent.
                                             P-Value is greater than or equal to 0.05
                                             Fail to reject the null hypothesis (H₀)
                                             Consider removing or deprioritizing the feature.
                                    Alternative Hypothesis (H₁):
                                             There is an association between the two categorical variables.
                                             In other words, the categorical variables are not independent.
                                             P-Value is less than 0.05
                                             Reject the null hypothesis (H₀)
                                             Consider keeping or prioritizing the feature.

                                             "

                        Breifly explain these analysis in readable and table format for easy analysis and headings.
                        """
                 response6=llm.invoke(cat_vs_cat_prompt) 
                 st.info(response6.content) 

      #------------------------------------------------------------------------------------------------------

            st.title("Data Cleaning:")
            nan_70_option=st.radio('Do you want to remove rows and columns with more than 70% missing values?',
                                   ('Select an option','Yes','No' ))
            
            
            
            #In Pandas:
               #df[mask] —> filters rows.
               #df.loc[:, mask] —> filters columns.
            
            
            df_removed=df.copy()
            if nan_70_option =='Yes':
                threshold=0.7
                df_removed=df_removed[df_removed.isnull().mean(axis=1) < threshold]
                df_removed=df_removed.reset_index(drop=True)
                print(df_removed.isnull().mean(axis=0))
                df_removed=df_removed.loc[:,df_removed.isnull().mean(axis=0) < threshold]
                st.write("You chose to remove rows with >70% missing values.")
                st.dataframe(df_removed.isnull().sum())

            elif nan_70_option == 'No':
                st.write("You chose not to remove those rows.")
                st.dataframe(df_removed.isnull().sum())    

            if nan_70_option in ['Yes','No']:

               nan_num_option = st.radio(
               "Choose method to fill missing values of numerical column",
               ( 'Select an option',
               "Fill with mean",
               "Fill with median", 
               "Fill with zero",
                 "Forward fill",
                 'Backward fill')
                              )
               df_cleaned = df_removed.copy()
               df_cleaned_num=df_cleaned[[feature for feature in df_cleaned.columns if df_cleaned[feature].dtype != 'O']]
               print(df_cleaned_num)


            

               if nan_num_option == "Fill with mean":
                  df_cleaned_num = df_cleaned_num.fillna(df.mean(numeric_only=True))

               elif nan_num_option == "Fill with median":
                  df_cleaned_num = df_cleaned_num.fillna(df.median(numeric_only=True))

               elif nan_num_option == "Fill with zero":
                  df_cleaned_num = df_cleaned_num.fillna(0)

               elif nan_num_option == "Forward fill":
                  df_cleaned_num = df_cleaned_num.fillna(method="ffill").fillna(method='bfill')

               elif nan_num_option == "Backward fill":
                  
                  df_cleaned_num = df_cleaned_num.fillna(method="bfill").fillna(method="ffill")

               st.dataframe(df_cleaned_num .isnull().sum().T) 

               if nan_num_option in [ "Fill with mean",
               "Fill with median", 
               "Fill with zero",
                 "Forward fill",
                 'Backward fill'] :
                    
                  nan_cat_option = st.radio(
               "Choose method to fill missing values of categorical column",
               ( 'Select an option',
               "Fill with mode",
                "Forward fill",
                 'Backward fill'))
                  df_cleaned_cat=df_cleaned[[feature for feature in df_cleaned.columns if df_cleaned[feature].dtype == 'O']]
                  if nan_cat_option == "Fill with mode":
                     df_cleaned_cat = df_cleaned_cat.fillna(df_cleaned_cat.mode())
                  elif nan_cat_option == "Forward fill":
                     df_cleaned_cat = df_cleaned_cat.fillna(method="ffill").fillna(method='bfill')
                  elif nan_cat_option == "Backward fill":
                     df_cleaned_cat = df_cleaned_cat.fillna(method="bfill").fillna(method="ffill")

                  st.dataframe(df_cleaned_cat.isnull().sum()) 





          #new_cleaned_df=pd.concat([df_cleaned_num, df_cleaned_cat], axis=1)

         #------------------------------------------------------------------------------------------------------
                  if nan_cat_option in ["Fill with mode","Forward fill",'Backward fill']:
                     new_cleaned_df=pd.concat([df_cleaned_num, df_cleaned_cat], axis=1)

             

            
  




            
            


   


            #option = st.selectbox(
                    #"select the target column",
                    #[column for column in df.columns],index=None
                                    #)   
            #st.write("You selected:", option)
                            

                        
                 
             

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        except Exception as e:
            st.error(e )          
        




















else:
    st.info("Please upload a structured data file to begin.") 




