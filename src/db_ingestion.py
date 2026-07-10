import pandas as pd
from sqlalchemy import create_engine,text
from config.database import DATABASE_URL,DB_NAME,DB_USER,DB_PASSWORD,DB_HOST,DB_PORT

def initialize_database():

    # Connects to Mysql to verify/create the database instance, loading the raw 
    # Consumer data from CSV , prepare aauthentication and system table

    base_url=f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}"
    server_engine=create_engine(base_url)
    print(f"[!] Ensuring MySQL Database {DB_NAME} exist.....")

    with server_engine.connect() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}"))
        conn.commit()

    engine=create_engine(DATABASE_URL)
    csv_path="data/raw/data_ecommerce_customer_churn.csv"
    print(f"Reading raw dataset from {csv_path}")
    try:
        df=pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR Could not locate the file")
        return
    
    print("Uploading raw customer data into raw_customer_churn table ")
    df.to_sql("raw_customer_churn_data",engine,if_exists="replace",index=False, chunksize=1000)
    print("[Success] Data successfully pushed to mysql")

    # Build explicit user credential data mappings for the login portal
    print("[!] Seeding authentication tracking systems......")
    mock_user=pd.DataFrame([
        {"username":"admin_director","password":"password123","role":"admin"},
        {"username":"agent_success","password":"password321","role":"staff"}])
    
    mock_user.to_sql("users",engine,if_exists="replace",index=False)

    # Create structural log records tracking user queriesand live predictions
    print("[!] Engineering telemetry logging mechanism....")
    with engine.connect() as conn:
        conn.execute(text("""CREATE TABLE IF NOT EXISTS model_prediction_history(
                          id INT AUTO_INCREMENT PRIMARY KEY,
                          tenure INT,
                          warehouse_to_home INT,
                          complain INT,
                          cashback_amount DOUBLE,
                          churn_probability DOUBLE,
                          prediction_output INT, 
                          operator_username VARCHAR(255),
                          execution_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
                        """))
        #Create model registry table ---------> fulfilling the project criteria without bloat
        conn.execute(text("""CREATE TABLE IF NOT EXISTS model_registry(
                          version_id INT AUTO_INCREMENT PRIMARY KEY,
                          model_name VARCHAR(255),
                          accuracy_score DOUBLE,
                          saved_file_path VARCHAR(225), training_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
                        """))
        conn.commit()
    print("[!Success] Database structural initialization completed.....")






