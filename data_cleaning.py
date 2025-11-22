import pandas as pd 




def main():
    
    ####Checking cusotmer id 


    df = pd.read_csv("data/telco_customer_data_v2.csv")
    rename_dict = {
    "customerID": "customer_id",
    "gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churn"
}   
    

    #_______________________________Helper print functions__________________________________
    def autoprintb4(column):
        print(f"Before cleaning: unique entries in {column} column: {df[column].unique()}") 
    def autoprintaft(column):
        print(f"After cleaning: Unique entries in {column} column: {df[column].unique()}") 
    #
    df = df.rename(columns = rename_dict)
    
    len_criteria = len("CUST00001")
    wrong_len_count=0
    wrong_index_count = 0
    for entry,index in zip(df["customer_id"],df.index):
        if len(entry) != len_criteria:
            wrong_len_count +=1 
        entry =entry.lstrip("CUST")
        entry = int(entry)
        if entry-1 != index:
            wrong_index_count += 1


    print(f"Number of misswritten entries: {wrong_len_count}")
    print(f"Number of misplaced entries: {wrong_index_count}")
   
    autoprintb4('gender')
    df["gender"] = df["gender"].apply(
    lambda x: 1 if isinstance(x, str) and x.lower().startswith("m")
    else (0 if isinstance(x, str) and x.lower().startswith("f") 
          else x))
    
    autoprintaft("gender")

    
    autoprintb4('senior_citizen')
    no_criteria = ["No","not senior"]
    df["senior_citizen"] = df["senior_citizen"].apply(
        lambda x:1 if isinstance(x,str) and x == "Yes" 
        else 0 if isinstance(x,str) and x in no_criteria
        else x
    )
    df["senior_citizen"] = pd.to_numeric(df["senior_citizen"], errors="coerce") #to get the str ones and zeroes        
    autoprintaft('senior_citizen')
    
 
 
    

    def yes_no_encoder(column):
        print(f"Unique entries in {column} column:{df[column].unique()}")
        df[column] = df[column].apply(
        lambda x:1 if isinstance(x,str) and x =="Yes"
        else 0 if isinstance(x,str) and x =="No"
        else x)
        print(f"Unique entries in {column} column:{df[column].unique()} after cleaning")
    
    yes_no_encoder("partner")
    yes_no_encoder("dependents")
  
    print(df["multiple_lines"].unique())

    print(f"Unique entries in multiple_linescolumn:{df['multiple_lines'].unique()}")



    #__________________________________phone_service &  multiple_lines___________________________________________________________
    bad_rows =df[(df["phone_service"]=="Yes")&(df["multiple_lines"]=="No phone service")]
    yes_no_encoder("phone_service")
    print(f"How many rows where there are phone service = yes and no phone service in multiple lines column: {len(bad_rows)}")

    
    df["multiple_lines"] = df["multiple_lines"].apply(
        lambda x: 1 if x =="Yes" 
        else 0 if isinstance(x,str) and x.startswith("N") 
        else x)
    print(f"Unique entries in multiple_lines column:{df['multiple_lines'].unique()} after cleaning") 
    
    #Taking care of the assumed misplaced no phone service
    df.loc[bad_rows.index,"multiple_lines"] =0 
    df.loc[df["phone_service"] == 0, "multiple_lines"] = 0 #making sure no other inconstinces are happening by letting phone service take precedent

    print(f"Unique entries in multiple_lines column:{df['multiple_lines'].unique()} after also taking care of bad rows") 
    

    ##__________________________internet_service & online_security______________________________
    print(f"Unique entries in internet_service column: {df['internet_service'].unique()}") 

    #Encode 0 for no internet, 1 for DSL (its low-speed) and 2 for Fiber optic

    df["internet_service"]= df["internet_service"].apply(
        lambda x:0 if x == "No" 
        else 1 if x == "DSL" 
        else 2
    )
    print(f"Unique entries in internet_service column: {df['internet_service'].unique()} after cleaning") 


    
    #_________________________Fixing internet service "subcolumns" first
    columns_to_fix = ["online_security","online_backup","device_protection","tech_support","streaming_tv","streaming_movies"]

    bad_rows = df[(df["internet_service"]!=0)
                  & (df[columns_to_fix]=="No internet service").any(axis=1)
                  ]
    print(f"Number of rows where there is implied internet service but 'subcolumns' say no internet service:    {len(bad_rows)}")

    for col in columns_to_fix:
        df.loc[bad_rows.index,col] = 0  

    def subnetcleaner(col):
        df[col]=df[col].apply(
            lambda x : 1 if isinstance(x,str) and x.startswith("Y") 
            else 1 if x == "True"
            else 0 if isinstance(x,str) and x.startswith("N") 
            else x
        )
        

    for col in columns_to_fix:
        autoprintb4(col)
        subnetcleaner(col)
        autoprintaft(col)
    
    df.loc[df["internet_service"] == 0, columns_to_fix] = 0 #taking care of those where the opposite is true


    autoprintb4('contract')

    df['contract'] = df['contract'].apply(
        lambda x: 0 if x.startswith("M")
        else 1 if x == "One year"
        else 2 if x == "Two year"
        else x
    )
    autoprintaft('contract')
    autoprintb4('paperless_billing')
    yes_no_encoder('paperless_billing')
    autoprintaft('paperless_billing')



    ###Fixing
    autoprintb4('payment_method')
    #I think there is no natural order in payment method 
    #Rather a distinction between if a customer pays automatically or if they manually pay
    #If its automatic they will forget

    df['payment_method'] = df['payment_method'].apply(
        lambda x: 1 if isinstance(x,str) and x.endswith('(automatic)')
        else 0 if isinstance(x,str) and not x.endswith('(automatic)')
        else x
    )
    autoprintaft('payment_method')
    #__________checking numerical columns for str or na
    #autoprintb4('monthly_charges')


    df["total_charges"] = (
    df["total_charges"].astype(str).str.replace(r"[^0-9.]", "", regex=True)  # remove $, USD, commas, letters, etc.
    )

    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df["monthly_charges"] =pd.to_numeric(df["monthly_charges"], errors="coerce")

    autoprintb4("churn")

    df['churn'] = df['churn'].apply(
        lambda x: 1 if x.startswith("Y")
        or x.startswith("C")
        else 0 if x.startswith("N")
        else pd.NA if x.startswith("Unknown")
        else x)

    autoprintaft("churn") 
    
    df.to_csv("data/data_cleaned.csv")
    print("Cleaned data saved")
    print(df.dtypes)
    ####


  

    



    


        

    
main()



