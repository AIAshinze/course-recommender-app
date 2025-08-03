# 🎓 Course Recommendation System

Welcome to the Course Recommendation System — an interactive, Streamlit-based web app designed to help users discover online courses tailored to their interests using multiple machine learning and recommendation algorithms.

---

## 🚀 Features

- 🔍 **Course Similarity** – Recommends courses based on content similarity (BoW + Cosine Similarity)
- 🧑‍🎓 **User Profile Modeling** – Matches users to relevant courses using dot product with genre features
- 🧠 **Clustering & PCA** – Groups users into learning clusters to suggest popular courses
- 🤝 **Collaborative Filtering (KNN & NMF)** – Learns from other users’ behaviors to suggest new courses
- 🤖 **Neural Network Recommender** – Embedding-based recommendation using TensorFlow
- 📊 **Regression & Classification** – Predicts ratings and preferences using embedded features

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**: Python, pandas, scikit-learn, TensorFlow, Surprise
- **Machine Learning**: KMeans, PCA, KNN, NMF, Linear Regression, Random Forest
- **Data**: User-course ratings, course metadata, genre labels

---

## 📂 File Structure

```text
├── recommender_app.py          # Streamlit frontend
├── backend.py                  # All model logic
├── requirements.txt            # Dependencies
├── sim.csv                     # Course similarity matrix
├── ratings.csv                 # User-course rating matrix
├── course_processed.csv        # Course metadata
├── courses_bows.csv            # Bag-of-words for courses
└── README.md
