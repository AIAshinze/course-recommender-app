import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from surprise import Dataset, Reader, KNNBasic, NMF
import os  # Added for file existence checks

# List of models (removed unimplemented models)
models = (
    "Course Similarity",
    # "User Profile",  # Removed due to missing dependencies
    # "Clustering",    # Removed (not implemented)
    # "Clustering with PCA",  # Removed (not implemented)
    "KNN",
    "NMF",
    "Neural Network",
    # "Regression with Embedding Features",  # Removed from predict
    # "Classification with Embedding Features"  # Removed from predict
)

# Cache for trained models
backend_models = {}

# -------------- DATA LOADERS -------------- #

def load_ratings():
    if not os.path.exists("ratings.csv"):
        raise FileNotFoundError("ratings.csv not found")
    return pd.read_csv("ratings.csv")

def load_course_sims():
    if not os.path.exists("sim.csv"):
        raise FileNotFoundError("sim.csv not found")
    return pd.read_csv("sim.csv")

def load_courses():
    if not os.path.exists("course_processed.csv"):
        raise FileNotFoundError("course_processed.csv not found")
    df = pd.read_csv("course_processed.csv")
    df['TITLE'] = df['TITLE'].str.title()
    return df

def load_bow():
    if not os.path.exists("courses_bows.csv"):
        raise FileNotFoundError("courses_bows.csv not found")
    return pd.read_csv("courses_bows.csv")

# -------------- MAPPING HELPERS -------------- #
def add_new_ratings(new_courses):
    res_dict = {}
    if len(new_courses) > 0:
        ratings_df = load_ratings()
        new_id = ratings_df['user'].max() + 1
        users = [new_id] * len(new_courses)
        ratings = [3.0] * len(new_courses)
        res_dict['user'] = users
        res_dict['item'] = new_courses
        res_dict['rating'] = ratings
        new_df = pd.DataFrame(res_dict)
        updated_ratings = pd.concat([ratings_df, new_df])
        updated_ratings.to_csv("ratings.csv", index=False)
        return new_id

def get_doc_dicts():
    bow_df = load_bow()
    # Fixed dictionary creation
    idx_id_dict = bow_df.set_index('doc_index')['doc_id'].to_dict()
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    return idx_id_dict, id_idx_dict

# -------------- COURSE SIMILARITY -------------- #

def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    res = {}
    for enrolled_course in enrolled_course_ids:
        if enrolled_course not in id_idx_dict:
            continue
        idx1 = id_idx_dict[enrolled_course]
        for unselect_course in unselected_course_ids:
            if unselect_course not in id_idx_dict:
                continue
            idx2 = id_idx_dict[unselect_course]
            # Handle possible index out-of-bounds
            if idx1 < sim_matrix.shape[0] and idx2 < sim_matrix.shape[1]:
                sim = sim_matrix[idx1, idx2]
                if unselect_course not in res or sim > res[unselect_course]:
                    res[unselect_course] = sim
    return dict(sorted(res.items(), key=lambda item: item[1], reverse=True))

# -------------- USER PROFILE -------------- #
# Removed due to missing dependencies (load_user_profiles, load_course_genres, etc.)

# -------------- KNN & NMF (Surprise) -------------- #

def train_surprise_model(model_type, params):
    ratings = load_ratings()
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['user', 'item', 'rating']], reader)
    trainset = data.build_full_trainset()

    if model_type == "KNN":
        sim_options = {'name': 'cosine', 'user_based': False}
        model = KNNBasic(
            k=params.get('k', 20), 
            sim_options=sim_options,
            verbose=False  # Added to reduce output noise
        )
    elif model_type == "NMF":
        model = NMF(
            n_factors=params.get('n_factors', 32), 
            random_state=123
        )  # Removed unsupported init_low/init_high

    model.fit(trainset)
    backend_models[model_type] = model

def predict_surprise_model(model_type, user_id):
    model = backend_models.get(model_type)
    if not model:
        raise ValueError(f"Model {model_type} not trained")
    
    ratings = load_ratings()
    enrolled = ratings[ratings['user'] == user_id]['item'].tolist()
    all_items = ratings['item'].unique()
    predictions = {}
    
    for item in all_items:
        if item not in enrolled:
            try:
                pred = model.predict(user_id, item)
                predictions[item] = pred.est
            except Exception as e:
                print(f"Prediction failed for user {user_id}, item {item}: {str(e)}")
    return predictions

# -------------- NEURAL NETWORK -------------- #

class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_size=50):
        super().__init__()
        self.user_embedding = tf.keras.layers.Embedding(
            num_users, embedding_size, input_length=1)
        self.item_embedding = tf.keras.layers.Embedding(
            num_items, embedding_size, input_length=1)
        self.flatten = tf.keras.layers.Flatten()
        self.dot = tf.keras.layers.Dot(axes=1)

    def call(self, inputs):
        user_vec = self.flatten(self.user_embedding(inputs[0]))
        item_vec = self.flatten(self.item_embedding(inputs[1]))
        return self.dot([user_vec, item_vec])

def train_neural(params):
    ratings = load_ratings()
    # Create mappings for user and item IDs
    user_ids = ratings['user'].unique()
    item_ids = ratings['item'].unique()
    user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
    
    # Map to indices
    ratings['user_idx'] = ratings['user'].map(user_to_index)
    ratings['item_idx'] = ratings['item'].map(item_to_index)
    
    x = [ratings['user_idx'].values, ratings['item_idx'].values]
    y = ratings['rating'].values
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=123)
    
    num_users = len(user_ids)
    num_items = len(item_ids)

    model = RecommenderNet(num_users, num_items, 50)
    model.compile(
        optimizer='adam', 
        loss='mse', 
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    model.fit(
        x=x_train, 
        y=y_train,
        batch_size=params.get('batch_size', 64),
        epochs=params.get('epochs', 10),
        validation_data=(x_val, y_val),
        verbose=0
    )
    # Store model + mappings
    backend_models['neural'] = {
        'model': model,
        'user_map': user_to_index,
        'item_map': item_to_index
    }

# -------------- REGRESSION/CLASSIFICATION -------------- #
# Removed from predict flow (not implemented for recommendations)

# -------------- TRAIN DISPATCH -------------- #

def train(model_name, params=None):
    if params is None:
        params = {}
        
    if model_name == "KNN":
        train_surprise_model("KNN", params)
    elif model_name == "NMF":
        train_surprise_model("NMF", params)
    elif model_name == "Neural Network":
        train_neural(params)
    # Regression/classification models removed from predict flow

# -------------- PREDICT DISPATCH -------------- #

def predict(model_name, user_ids, params=None):
    if params is None:
        params = {}
        
    sim_threshold = params.get("sim_threshold", 60) / 100.0
    results = []

    if model_name == "Course Similarity":
        idx_id_dict, id_idx_dict = get_doc_dicts()
        sim_matrix = load_course_sims().to_numpy()
        ratings_df = load_ratings()
        
        for uid in user_ids:
            enrolled = ratings_df[ratings_df['user'] == uid]['item'].tolist()
            res = course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled, sim_matrix)
            for cid, score in res.items():
                if score >= sim_threshold:
                    results.append((uid, cid, score))

    elif model_name == "KNN":
        for uid in user_ids:
            res = predict_surprise_model("KNN", uid)
            for cid, score in res.items():
                results.append((uid, cid, score))

    elif model_name == "NMF":
        for uid in user_ids:
            res = predict_surprise_model("NMF", uid)
            for cid, score in res.items():
                results.append((uid, cid, score))

    elif model_name == "Neural Network":
        model_data = backend_models.get('neural')
        if not model_data:
            raise ValueError("Neural Network model not trained")
            
        model = model_data['model']
        user_map = model_data['user_map']
        item_map = model_data['item_map']
        ratings = load_ratings()
        all_items = ratings['item'].unique()
        
        for uid in user_ids:
            # Skip unknown users
            if uid not in user_map:
                continue
                
            enrolled = ratings[ratings['user'] == uid]['item'].tolist()
            # Get valid items with mappings
            valid_items = [item for item in all_items 
                         if item not in enrolled and item in item_map]
            
            if not valid_items:
                continue
                
            # Batch prediction
            user_idx = [user_map[uid]] * len(valid_items)
            item_idx = [item_map[item] for item in valid_items]
            scores = model.predict([np.array(user_idx), np.array(item_idx)], verbose=0).flatten()
            
            for cid, score in zip(valid_items, scores):
                results.append((uid, cid, score))

    return pd.DataFrame(results, columns=["USER", "COURSE_ID", "SCORE"]) if results else pd.DataFrame()