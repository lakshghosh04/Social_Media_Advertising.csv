import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

st.set_page_config(page_title="Social Media Ad Dashboard", layout="wide")
st.title("📊 Social Media Advertising Insights + ROI Prediction")

# ==== 1. Ask user to upload dataset ====
uploaded_file = st.file_uploader("Upload your Social Media Advertising CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")
    st.write("Preview of data:")
    st.dataframe(df.head())

    # ==== 2. Train Model ====
    features = ["Channel_Used", "Target_Audience", "Campaign_Goal", "Impressions", "Clicks"]
    target = "ROI"

    # Check if all columns exist
    if all(col in df.columns for col in features + [target]):
        df_model = df[features + [target]].dropna()

        X = df_model[features]
        y = df_model[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        categorical_cols = ["Channel_Used", "Target_Audience", "Campaign_Goal"]
        numeric_cols = ["Impressions", "Clicks"]

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
                ("num", "passthrough", numeric_cols)
            ]
        )

        model_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        model_pipeline.fit(X_train, y_train)
        joblib.dump(model_pipeline, "roi_predictor.pkl")

        # ==== 3. Sidebar Prediction Tool ====
        st.sidebar.header("🔮 ROI Prediction Tool")
        with st.sidebar.form("prediction_form"):
            channel = st.selectbox("Channel Used", df["Channel_Used"].unique())
            audience = st.selectbox("Target Audience", df["Target_Audience"].unique())
            goal = st.selectbox("Campaign Goal", df["Campaign_Goal"].unique())
            impressions = st.number_input("Expected Impressions", min_value=0)
            clicks = st.number_input("Expected Clicks", min_value=0)
            submit = st.form_submit_button("Predict ROI")

        if submit:
            model = joblib.load("roi_predictor.pkl")
            input_df = pd.DataFrame([{
                "Channel_Used": channel,
                "Target_Audience": audience,
                "Campaign_Goal": goal,
                "Impressions": impressions,
                "Clicks": clicks
            }])
            prediction = model.predict(input_df)[0]
            st.sidebar.success(f"Predicted ROI: {prediction:.2f}")

    else:
        st.error("❌ Uploaded dataset does not have the required columns: " + ", ".join(features + [target]))
else:
    st.info("📥 Please upload a CSV file to start.")
