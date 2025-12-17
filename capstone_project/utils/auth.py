# utils/auth.py
import pandas as pd

# Load customer data ONCE
df_auth = pd.read_csv("data/recommendation_dataset.csv")

# Clean columns
df_auth.columns = df_auth.columns.str.strip()

def authenticate_customer(customer_id, dob):
    """
    Authenticate customer using CustomerID and DOB
    """
    customer = df_auth[
        (df_auth["CustomerID"] == customer_id) &
        (df_auth["CustomerDOB"] == dob)
    ]

    if customer.empty:
        return False

    return True
