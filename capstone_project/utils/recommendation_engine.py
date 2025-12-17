import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

# LOAD DATASET (FORCE STRING + CLEAN)
df = pd.read_csv("data/recommendation_dataset.csv", dtype=str)
df.columns = df.columns.str.strip()
df['CustomerID'] = df['CustomerID'].str.strip().str.upper()

# Convert numeric columns properly
for col in df.columns:
    if col != 'CustomerID':
        df[col] = pd.to_numeric(df[col], errors='ignore')

# FEATURE SELECTION
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

remove_cols = [
    'CustomerID',
    'Loan_type',
    'credit_cardtype',
    'investment_type',
    'savings_plan_type'
]

for col in remove_cols:
    if col in numeric_features:
        numeric_features.remove(col)
    if col in categorical_features:
        categorical_features.remove(col)

# PREPROCESSING
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# KMEANS CLUSTERING
X = preprocessor.fit_transform(df[numeric_features + categorical_features])
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# -------------------------------
# RECOMMENDATION CORE
# -------------------------------
def recommend_from_cluster(customer_id, product_col, top_n=3):
    """Get top N recommendations for a specific customer only."""
    customer_id = str(customer_id).strip().upper()
    customer = df[df['CustomerID'] == customer_id]

    if customer.empty:
        return None

    cluster_id = customer['cluster'].values[0]
    cluster_data = df[df['cluster'] == cluster_id]

    recommendations = (
        cluster_data[product_col]
        .value_counts()
        .dropna()
        .head(top_n)
        .index
        .tolist()
    )

    return recommendations

# -------------------------------
# CHAT ENTRY FUNCTION
# -------------------------------
def get_recommendation(user_input, logged_in_customer_id=None):
    """Get recommendations for one or multiple products for the logged-in customer (formatted with bullets)."""
    if logged_in_customer_id is None:
        return "🔒 Please login as customer to get recommendations."

    text = user_input.lower()

    product_map = {
        "loan": "Loan_type",
        "credit": "credit_cardtype",
        "investment": "investment_type",
        "saving": "savings_plan_type"
    }

    results = []

    for key, col in product_map.items():
        if key in text:
            recs = recommend_from_cluster(logged_in_customer_id, col)
            if recs:
                bullet_list = "\n".join([f"• {r}" for r in recs])
                results.append(f"**{key.title()} Recommendations:**\n{bullet_list}")

    if not results:
        return "Please mention loan, credit card, investment, or savings."

    # Prefix with Customer ID
    return f"Customer ID : {logged_in_customer_id}\n\n" + "\n\n".join(results)
