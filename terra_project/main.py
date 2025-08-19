from fastapi import FastAPI
import pandas as pd

app = FastAPI()

def return_farmer_outputs(phone_number):
    phone_number = int(phone_number)
    df = pd.read_csv(
        "https://raw.githubusercontent.com/ayanatherate/TerraCred-AI/refs/heads/main/data/Data_with_phone_numbers.csv"
    )
    df['Farmer_Phone_Number'] = [int(str(i)[2:]) for i in df['Farmer_Phone_Number'].values]

    try:
        df_col = df.loc[df['Farmer_Phone_Number'] == int(phone_number)]
        cred_lim = int(df_col['credit_limit'].values[0])
        loan_approved = int(df_col['loan_approval'].values[0])

        if loan_approved == 1:
            return {
                "phone_number": phone_number,
                "status": "approved",
                "credit_limit": cred_lim,
                "loan_approval": 1,
                "top_positive_factors": [
                    "High NDVI (healthy crops)",
                    "Excellent repayment history",
                    "Irrigation access",
                    "PM-Kisan beneficiary"
                ],
                "risk_factors": [
                    "High price volatility",
                    "Distance to market"
                ]
            }
        else:
            return {
                "phone_number": phone_number,
                "status": "rejected",
                "credit_limit": cred_lim,
                "loan_approval": 0,
                "limiting_factors": [
                    "Past loan default",
                    "Poor soil nutrients",
                    "High yield variability",
                    "No irrigation access"
                ],
                "improvement_suggestions": [
                    "Soil testing and fertilization",
                    "Crop insurance enrollment",
                    "Join local FPO"
                ]
            }
    except:
        return {
            "phone_number": phone_number,
            "status": "not_found",
            "credit_limit": None,
            "loan_approval": None
        }


# ✅ API 1: GET with phone number in path
@app.get("/api/loan-status/{phone_number}")
async def api_get_loan_status(phone_number: str):
    return return_farmer_outputs(phone_number)


# ✅ API 2: POST with phone number in JSON body
@app.post("/api/loan-status")
async def api_post_loan_status(payload: dict):
    phone_number = payload.get("phone_number")
    return return_farmer_outputs(phone_number)
