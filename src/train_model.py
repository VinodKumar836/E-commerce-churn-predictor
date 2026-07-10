from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sqlalchemy import create_engine,text
from config.database import DATABASE_URL
import joblib
import os
def execute_model_training(X_train,X_test,Y_train,Y_test):
    print("Initialize hyperparameter tuning via RandomizedSearchCV")
    rf_base=RandomForestClassifier(random_state=42)

    # Define a clear parameter grid to search over
    param_distributions = {'n_estimators':[100,200,300,400,500], 'max_depth':[5,10,15,20,None], 'min_samples_split':[2,5,10],'min_samples_leaf':[1,2,4],'bootstrap':[True,False]}

    # Set up randomized search witb 3 fold cross validation\
    rf_random=RandomizedSearchCV(estimator=rf_base,param_distributions=param_distributions,n_iter=10,cv=3,random_state=42,n_jobs=-1,scoring='accuracy')
    print("[!!!] Searching for the optimal hyperparameter settings across the folds...... ")
    rf_random.fit(X_train,Y_train)

    # Extracting the optimal model configuration
    best_model=rf_random.best_estimator_
    print(f"[SUCCESS] Optimal parameters identified: {rf_random.best_params_}")

    # Evaluating the best model
    predictions=best_model.predict(X_test)
    accuracy=accuracy_score(Y_test,predictions)
    print(f"[SUCCESS] Tuned model accuracy achieved: {accuracy:.4f}")
    print("\n...... Classification Performance Report ......")
    print(classification_report(Y_test,predictions))

    # Persist physical binary model artifactsafely inside the models folder
    model_dir="models"
    os.makedirs(model_dir,exist_ok=True)
    model_path=os.path.join(model_dir,"random_forest_model.pkl")
    joblib.dump(best_model,model_path)
    print(f"Model saved to the path {model_path}")

    # Log model training execution results to your MySQL Server
    engine=create_engine(DATABASE_URL)
    with engine.connect() as conn:
        conn.execute(
            text("INSERT INTO model_registry(model_name,accuracy_score,saved_file_path)" \
            "VALUES (:name,:score,:path)"),{"name": "Random_Forest_Tuned_v1","score":float(accuracy),"path":model_path}
            )
        conn.commit()
    print("[SUCCESS] Model training metadata successfully logged in database registry")
    

