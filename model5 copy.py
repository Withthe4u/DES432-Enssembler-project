from csv import reader

from processing import allData, get_candidate_movies
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, time, gc

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from surprise import Dataset, KNNBasic, Reader, SVD
import pandas as pd

#import data
data = allData()

#check data that improted
'''
print(len(data["train"]))
print(len(data["val"])) 
'''

#import data to dataframe
train = data["train"]
val= data["val"]
movies= data["movies"] #unfinished movies data


all_movies = list(train["movie_id"].unique())
all_users  = list(train["user_id"].unique())
print("")


#[1]construct utility matrix
print("==========[1]constructing utility matrix==========")
print("========================================")

movie_to_idx = {m: i for i,m in enumerate(all_movies)} #make to dictionary
user_to_idx  = {u: i for i,u in enumerate(all_users)}
NumUsers, NumMovies = len(all_users),len(all_movies)
print(f"Number of users: {NumUsers}")
print(f"Number of movies: {NumMovies}") 


# Make a R-sparse utility matrix (User and Movie)
rows = train["user_id"].map(user_to_idx).values #map and then make an array
cols = train["movie_id"].map(movie_to_idx).values 
#print(np.dtype(train["rating"].values[1])) #int64
vals = train["rating"].values.astype(np.float32) #make an array and then change thw type to float 32 for faster computation
R_sparse = csr_matrix((vals, (rows, cols)), shape=(NumUsers, NumMovies)) #Compressed Sparse Row

print("========================================")
# Helper dicts
train_user_watched = data["train_user_watched"] 
val_user_movie     =  data["val_user_movie"] #validation of users
missingRate = 1 - R_sparse.nnz / (NumUsers * NumMovies) #nnz is non-zero things in matrix
print(f"Utility matrix : {R_sparse.shape}  ({R_sparse.nnz:,} observed ratings)")
print(f"Missing rate   : {missingRate:.4f}  ({missingRate*100:.1f}% empty)") 
print(f"Validation users : {len(val_user_movie)}")
print("")


#[2]precision@10 calculation
print("==========[2]precision calculation==========")

def precision_at_10(score,val_user_movie,train_user_watched,all_movies,movie_to_idx):
    hits = 0
    for userId, relevant in val_user_movie.items():
        scores = score(userId).copy().astype(float) #array of scores for all movies ofuser
        for m in train_user_watched.get(userId, set()): #get from dict ,if not fill with empty set
            idx = movie_to_idx.get(m) # make movie that have never seen to be in top 10
            if idx is not None: 
                scores[idx] = -np.inf #-infinity --> so it will not be in the top 10 
        top10_idx = np.argsort(scores)[-10:][::-1]
        top10 = [all_movies[i] for i in top10_idx]
        if relevant in top10:
            hits +=1
    return hits / len(val_user_movie) #R^unionR/R^


import pandas as pd
import numpy as np
from surprise import Dataset, Reader, KNNBasic

from processing import allData  
data = allData()

train              = data['train']
val_user_movie     = data['val_user_movie']
train_user_watched = data['train_user_watched']
all_movies         = data['all_movies']

# =========================
# Build Surprise dataset
# =========================
reader = Reader(rating_scale=(1, 5))
surprise_data = Dataset.load_from_df(train[['user_id','movie_id','rating']], reader)
trainset = surprise_data.build_full_trainset()

# =========================
# FAST candidate sampling
# =========================
import random

def get_candidates(train_user_watched, val_user_movie, all_movies, sample_size=300):
    user_candidates = {}
    for uid in val_user_movie.keys():
        seen = train_user_watched.get(uid, set())
        candidates = [m for m in all_movies if m not in seen]

        if len(candidates) > sample_size:
            candidates = random.sample(candidates, sample_size)

        user_candidates[uid] = candidates
    return user_candidates

user_candidates = get_candidates(train_user_watched, val_user_movie, all_movies)

# =========================
# Get scores (FAST)
# =========================
def get_scores(algo, user_candidates):
    scores = {}
    for uid, candidates in user_candidates.items():
        scores[uid] = {}
        for mid in candidates:
            scores[uid][mid] = algo.predict(uid, mid).est
    return scores

# =========================
# Top-N
# =========================
import heapq

def get_top_n(scores, n=10):
    top_n = {}
    for uid, movie_scores in scores.items():
        top_items = heapq.nlargest(n, movie_scores.items(), key=lambda x: x[1])
        top_n[uid] = [mid for mid, _ in top_items]
    return top_n

# =========================
# Precision@10
# =========================
def precision_at_10(top_n, val_user_movie):
    hits = 0
    for uid, gt in val_user_movie.items():
        if gt in top_n.get(uid, []):
            hits += 1
    return (hits / len(val_user_movie)) / 10


# =========================
# 🔵 USER-BASED CF
# =========================
print("Training User-Based CF...")

user_cf = KNNBasic(
    k=40,
    sim_options={'name': 'cosine', 'user_based': True},
    verbose=False
)
user_cf.fit(trainset)

scores_user = get_scores(user_cf, user_candidates)
top_n_user = get_top_n(scores_user)
p10_user = precision_at_10(top_n_user, val_user_movie)

print(f"User-Based CF P@10: {p10_user:.4f}")


# =========================
# 🟢 ITEM-BASED CF
# =========================
print("Training Item-Based CF...")

item_cf = KNNBasic(
    k=40,
    sim_options={'name': 'cosine', 'user_based': False},
    verbose=False
)
item_cf.fit(trainset)

scores_item = get_scores(item_cf, user_candidates)
top_n_item = get_top_n(scores_item)
p10_item = precision_at_10(top_n_item, val_user_movie)

print(f"Item-Based CF P@10: {p10_item:.4f}")

# =========================
# Train SVD
# =========================
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(train[['user_id','movie_id','rating']], reader)
trainset = dataset.build_full_trainset()

print('Training SVD...')
svd = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
svd.fit(trainset)
def get_top_n2(algo, train_user_watched, val_user_movie, all_movies, n=10):
    """
    For each user in val, predict scores for all UNSEEN movies
    using train_user_watched (from your groupData function),
    then return top-n movie_ids.

    Returns dict: {user_id: [movie_id1, ..., movie_id10]}
    """
    top_n = {}

    for uid in val_user_movie.keys():
        # Get unseen movies for this user (same logic as your get_candidate_movies)
        seen = train_user_watched.get(uid, set())
        candidates = [m for m in all_movies if m not in seen]

        # Predict rating for each unseen movie
        predictions = [(mid, algo.predict(uid, mid).est) for mid in candidates]
        predictions.sort(key=lambda x: x[1], reverse=True)

        top_n[uid] = [mid for mid, _ in predictions[:n]]

    return top_n


top_n_svd = get_top_n2(svd, train_user_watched, val_user_movie, all_movies)
p10_svd = precision_at_10(top_n_svd, val_user_movie)
print(f'SVD Precision@10: {p10_svd:.4f}')
# =========================
# Summary
# =========================
print("\n=== Summary ===")
print(f"User-Based CF  : {p10_user:.4f}")
print(f"Item-Based CF  : {p10_item:.4f}")

#[3]model1 user based collaborative filtering
print("==========[3]model1 user-based collaborative filtering==========")



#[4]model2 item based collaborative filtering 
print("==========[4]model2 item-based collaborative filtering==========")




#[5]model3 SVD
print("==========[5]model3 SVD==========")





#[6]ensemble model
print("==========[6]ensemble model==========")
def get_ensemble_scores(uid, mid):
    r1 = user_cf.predict(uid, mid).est   # UserCF
    r2 = item_cf.predict(uid, mid).est   # ItemCF
    r3 = svd.predict(uid, mid).est       # SVD
    return r1, r2, r3


def ensemble_score(uid, candidates, wu, wi, ws):
    scores = {}

    for mid in candidates:
        r1, r2, r3 = get_ensemble_scores(uid, mid)
        scores[mid] = wu*r1 + wi*r2 + ws*r3

    return scores

import heapq

def get_top_n_ensemble(user_candidates, wu, wi, ws, n=10):
    top_n = {}

    for uid, candidates in user_candidates.items():
        scores = ensemble_score(uid, candidates, wu, wi, ws)

        top_items = heapq.nlargest(n, scores.items(), key=lambda x: x[1])
        top_n[uid] = [mid for mid, _ in top_items]

    return top_n

best_score = 0
best_w = None

weights = np.arange(0, 1.1, 0.1)

for wu in weights:
    for wi in weights:
        ws = 1 - wu - wi

        if ws < 0:
            continue

        top_n = get_top_n_ensemble(user_candidates, wu, wi, ws)
        score = precision_at_10(top_n, val_user_movie)

        if score > best_score:
            best_score = score
            best_w = (wu, wi, ws)

print("Best weights:", best_w)
print("Best P@10:", best_score)
#==================================
print("\n==========[7]precision comparison==========")
