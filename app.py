# ==== MODEL TRAINING (Run once on app start) ====
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Select features and target
features = ["Channel_Used", "Target_Audience", "Campaign_Goal", "Impressions", "Clicks"]
target = "ROI"

# Drop missing values
df_model = df[features + [target]].dropna()

# Train-test split
X = df_model[features]
y = df_model[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess: One-hot encode categorical columns
categorical_cols = ["Channel_Used", "Target_Audience", "Campaign_Goal"]
numeric_cols = ["Impressions", "Clicks"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# Create pipeline
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train
model_pipeline.fit(X_train, y_train)

# Save model
joblib.dump(model_pipeline, "roi_predictor.pkl")

# ==== DASHBOARD INPUT & PREDICTION ====
st.sidebar.header("ROI Prediction Tool")
with st.sidebar.form("prediction_form"):
    channel = st.selectbox("Channel Used", df["Channel_Used"].unique())
    audience = st.selectbox("Target Audience", df["Target_Audience"].unique())
    goal = st.selectbox("Campaign Goal", df["Campaign_Goal"].unique())
    impressions = st.number_input("Expected Impressions", min_value=0)
    clicks = st.number_input("Expected Clicks", min_value=0)
    submit = st.form_submit_button("Predict ROI")

if submit:
    # Load model
    model = joblib.load("roi_predictor.pkl")
    # Prepare input
    input_df = pd.DataFrame([{
        "Channel_Used": channel,
        "Target_Audience": audience,
        "Campaign_Goal": goal,
        "Impressions": impressions,
        "Clicks": clicks
    }])
    # Predict
    prediction = model.predict(input_df)[0]
    st.sidebar.success(f"Predicted ROI: {prediction:.2f}")
