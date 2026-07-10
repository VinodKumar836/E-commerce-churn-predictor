import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config.database import DATABASE_URL
import joblib
import os

def extract_and_preprocess_data():
    # Pulling data directly from MySQL tables, cleans null dataset
    target_column="Churn"
    engine=create_engine(DATABASE_URL)
    print("Extracting data tables directly from MySQL......")
    df=pd.read_sql("SELECT * FROM raw_customer_churn_data",engine)

    if df.empty:
        raise ValueError("The 'raw_customer_churn_data' table is empty. Run db_ingestion first"
            )
    
    df[target_column]=pd.to_numeric(df[target_column],errors="coerce").fillna(0).astype(int)
    
    # Fill the numeric missing fields with median value -----> Imputation
    numeric_cols=df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Encoding the non numerical string data column to integer representation
    encoders={}
    categorical_cols=df.select_dtypes(include=["object"]).columns

    # Making directories 
    os.makedirs('models',exist_ok=True)

    print("Encoding categorical structural variables..........")
    for col in categorical_cols:
        le=LabelEncoder()
        df[col]=le.fit_transform(df[col].astype(str))
        encoders[col]=le

    # Save the encoder mapping pipline to use during live dashboard querying
    joblib.dump(encoders,"models/label_encoders.pkl")
    print("Saved data preprocessing encoders @ models/label_encoders.pkl")

    # formulate feature array(X) and target array(Y)
    drop_cols=target_column
    X=df.drop(columns=drop_cols)
    Y=df[target_column]

    # Saving tracking array layout column headers order so model interface input map correctly
    feature_columns=list(X.columns)
    joblib.dump(feature_columns,"models/feature_columns.pkl")

    # Generating split for training and testing dataset
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
    return X_train,X_test,Y_train,Y_test


