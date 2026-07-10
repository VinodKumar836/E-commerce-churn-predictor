# app/main_api.py
from fastapi import FastAPI, Form, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, text
import joblib
import os
import pandas as pd
from config.database import DATABASE_URL

# Initialize the main FastAPI application instance
app = FastAPI(title="E-Commerce Churn Mitigation Portal")

# SYSTEM MODIFICATION: Mount the local static folder to serve Tailwind CSS completely offline
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Setup HTML rendering capabilities using Jinja2 Templates
templates = Jinja2Templates(directory="app/templates")

# Initialize database connections and load ML model assets
engine = create_engine(DATABASE_URL)

try:
    model = joblib.load("models/random_forest_model.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    print("[SUCCESS] FastAPI successfully loaded model binaries from disk.")
except FileNotFoundError:
    print("[ERROR] Model files missing. Make sure to run 'python run_pipeline.py' first.")

# --- ROUTE 1: ENTRY POINT (LOGIN SCREEN VIEW) ---
@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request, error: str = None):
    """Renders the secure, premium login user interface entrance."""
    return templates.TemplateResponse(request=request,name="login.html", context={"error":error})

# --- ROUTE 2: LOGIN AUTHENTICATION ROUTER ---
@app.post("/login")
async def handle_login(username: str = Form(...), password: str = Form(...)):
    """Validates user credentials directly against rows stored in the MySQL users table."""
    with engine.connect() as conn:
        query = text("SELECT role FROM users WHERE username = :user AND password = :pass")
        result = conn.execute(query, {"user": username, "pass": password}).fetchone()
        
        if result:
            user_role = result[0]  # Extract string role ('admin' or 'staff') from row tuple
            # Route users dynamically based on their database structural access role permissions
            if user_role == "admin":
                return RedirectResponse(url=f"/dashboard/admin?username={username}", status_code=303)
            else:
                return RedirectResponse(url=f"/dashboard/staff?username={username}", status_code=303)
        else:
            return RedirectResponse(url="/?error=Invalid+credentials,+please+try+again.", status_code=303)

# --- ROUTE 3: ADMINISTRATIVE EXECUTIVE VIEW ---
@app.get("/dashboard/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request, username: str):
    """Queries aggregate metric structures straight from MySQL logging tables."""
    with engine.connect() as conn:
        # Fetch summary counting logs for system visualization
        total_queries_query = text("SELECT COUNT(*) FROM model_prediction_history")
        total_queries = conn.execute(total_queries_query).scalar() or 0
        
        high_risk_query = text("SELECT COUNT(*) FROM model_prediction_history WHERE prediction_output = 1")
        high_risk_count = conn.execute(high_risk_query).scalar() or 0
        
        # Pull live query history rows to pass directly to database tracking templates
        logs_query = text("""
            SELECT id, tenure, complain, churn_probability, prediction_output, execution_time 
            FROM model_prediction_history 
            ORDER BY execution_time DESC LIMIT 5
        """)
        recent_logs = conn.execute(logs_query).fetchall()
        
    return templates.TemplateResponse(
        request=request,
        name="admin_dashboard.html",
        context={
            'username':username,
            'total_queries':total_queries,
            'high_risk_count':high_risk_count,
            'recent_logs':recent_logs
            
        }
    )

# --- ROUTE 4: CLIENT AGENT STAFF VIEW ---
@app.get("/dashboard/staff", response_class=HTMLResponse)
async def staff_dashboard(request: Request, username: str, search_id: str = None, prediction_msg: str = None, probability: float = None):
    """Renders the AI Agent search hub and simulation playground interfaces."""
    customer_profile = None
    
    if search_id:
        # Pull a specific raw consumer timeline segment out of our operational database based on Tenure column
        with engine.connect() as conn:
            query = text("SELECT * FROM raw_customer_churn_data WHERE Tenure = :id LIMIT 1")
            result = conn.execute(query, {"id": search_id}).fetchone()
            
            if result:
                # Turn the record profile columns into an editable dictionary layout matching HTML mapping rules
                columns_query = text("SHOW COLUMNS FROM raw_customer_churn_data")
                columns = [col[0] for col in conn.execute(columns_query).fetchall()]
                customer_profile = dict(zip(columns, result))
                
    return templates.TemplateResponse( 
        request=request,
        name="staff_dashboard.html",
        context={
        "username": username, 
        "profile": customer_profile,
        "search_id": search_id,
        "prediction_msg": prediction_msg,
        "probability": probability
    })

# --- ROUTE 5: LIVE ML MODEL PREDICTION & DATABASE LOGGING ROUTER ---
@app.post("/predict")
async def run_prediction(
    request: Request,
    username: str = Form(...),
    tenure: int = Form(...),
    warehouse_to_home: int = Form(...),
    complain: int = Form(...),
    cashback_amount: float = Form(...)
):
    """Executes live Random Forest inference, calculates metrics, and updates logs in MySQL."""
    
    # 1. Create a placeholder matching the exact shapes and order used during training
    input_data = {col: 0 for col in feature_columns}
    
    # 2. Update parameters with the values submitted from our frontend HTML forms
    # Mapping to your exact dataset column casing
    if 'Tenure' in input_data: input_data['Tenure'] = tenure
    if 'WarehouseToHome' in input_data: input_data['WarehouseToHome'] = warehouse_to_home
    if 'Complain' in input_data: input_data['Complain'] = complain
    if 'CashbackAmount' in input_data: input_data['CashbackAmount'] = cashback_amount
    
    # Convert input payload dictionary array cleanly to a Pandas dataframe structure
    input_df = pd.DataFrame([input_data])
    
    # 3. Process machine learning scoring calculations
    churn_prob_array = model.predict_proba(input_df) 
    churn_prob = float(churn_prob_array[0][1])  # Extract specific probability score of class 1 (churn)
    final_output = 1 if churn_prob >= 0.5 else 0
    
    # 4. Save audit metrics logs straight back down to the MySQL tracking layer
    with engine.connect() as conn:
        log_query = text("""
            INSERT INTO model_prediction_history 
            (tenure, warehouse_to_home, complain, cashback_amount, churn_probability, prediction_output, operator_username)
            VALUES (:tenure, :wh, :comp, :cash, :prob, :out, :user)
        """)
        conn.execute(log_query, {
            "tenure": tenure, "wh": warehouse_to_home, "comp": complain, "cash": cashback_amount,
            "prob": churn_prob, "out": final_output, "user": username
        })
        conn.commit()
        
    msg = "ALERT: High Risk Churn Candidate!" if final_output == 1 else "Safe: Retained Shopper."
    return RedirectResponse(
        url=f"/dashboard/staff?username={username}&prediction_msg={msg}&probability={round(churn_prob*100, 2)}", 
        status_code=303
    )

