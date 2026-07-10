from src.db_ingestion import initialize_database
from src.data_preprocessing import extract_and_preprocess_data
from src.train_model import execute_model_training

if __name__=="__main__":
    print("=============================================================")
    print("    STARTING END TO END E COMMERCE CHURN TRAINING RUN   ")
    print("=============================================================\n")

    # Step Initialise SQL instances and load your CSV data file
    print("[STEP 1/3] Launching database initialisation and ingestion.......")
    initialize_database()
    print("[SUCCESS] Step 1 finished cleanly.\n")

    # Querying database tables directly and structure vectors
    print("[STEP 2/3] Extracting ad transforming features from MySQL.......")
    X_train, X_test, Y_train, Y_test=extract_and_preprocess_data()
    print("[SUCCESS] Step 2 finished cleanly.\n")

    # Run RandomForestClassifier training and update MySQL logs
    print("[STEP 3/3] Commencing Random Forest classifier optimization.......")
    execute_model_training( X_train, X_test, Y_train, Y_test)

    print("=============================================================")
    print("               ALL THE OPERATIONS RUN SUCCESSFULLY           ")
    print("=============================================================\n")


    


