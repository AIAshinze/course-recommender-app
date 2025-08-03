# recommender_app.py

import streamlit as st
import pandas as pd
from backend import models, train, predict

# App Config
st.set_page_config(page_title="📚 Course Recommender", layout="wide")
st.title("📚 Course Recommendation System")

# Sidebar Model Selector
st.sidebar.header("Model Settings")
model_selection = st.sidebar.selectbox("Choose a Recommendation Model", models)

# Sidebar Hyperparameters
params = {}
if model_selection == "Course Similarity":
    params["sim_threshold"] = st.sidebar.slider("Similarity Threshold (%)", 0, 100, 60)
elif model_selection == "KNN":
    params["k"] = st.sidebar.slider("Number of Neighbors (k)", 1, 50, 20)
elif model_selection == "NMF":
    params["n_factors"] = st.sidebar.slider("Latent Dimensions", 4, 100, 32)
elif model_selection == "Neural Network":
    params["epochs"] = st.sidebar.slider("Epochs", 5, 50, 10)
    params["batch_size"] = st.sidebar.selectbox("Batch Size", [32, 64, 128], index=1)

# Train Model Button
if st.sidebar.button("Train Model"):
    with st.spinner("Training model..."):
        try:
            train(model_selection, params)
            st.success(f"✅ {model_selection} model trained successfully!")
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")

# User Input
st.subheader("🧑 Enter User ID(s)")
user_input = st.text_input("Enter user ID(s) (comma-separated):", "1")

# Generate Recommendations
if st.button("Generate Recommendations"):
    try:
        user_ids = [int(uid.strip()) for uid in user_input.split(",") if uid.strip().isdigit()]
        
        if not user_ids:
            st.warning("⚠️ Please enter valid user ID(s)")
            st.stop()
            
        with st.spinner("Generating recommendations..."):
            rec_df = predict(model_selection, user_ids, params)

        if rec_df.empty:
            st.warning("⚠️ No recommendations found. Try another user or model.")
        else:
            st.success(f"✅ Found {len(rec_df)} recommendations")
            st.dataframe(rec_df.sort_values("SCORE", ascending=False).reset_index(drop=True))

    except Exception as e:
        st.error(f"❌ Recommendation error: {str(e)}")

# Add some helpful information
st.sidebar.markdown("---")
st.sidebar.info("""
**User ID Notes:**
- Valid user IDs are positive integers
- New users are automatically created when you add ratings
- Default test user: `1`
""")

# Display sample data if needed
if st.sidebar.checkbox("Show Sample Data"):
    try:
        st.subheader("Sample Ratings Data")
        st.write(pd.read_csv("ratings.csv").head())
    except:
        st.warning("Couldn't load ratings data")