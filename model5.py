from processing import allData, get_candidate_movies
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, time, gc

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from surprise import Dataset, Reader, SVD
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
        top10 = [all_movies[i] for i in np.argsort(scores)[-10:]] #
        if relevant in top10:
            hits +=1
    return hits / len(val_user_movie) #R^unionR/R^


#[3]model1 user based collaborative filtering
print("==========[3]model1 user-based collaborative filtering==========")

#Step 3.1: User–User Cosine Similarity 
# normalize --> sim(ux,ui)
R_user_norm = normalize(R_sparse, norm='l2') #l2=itself/euclidean 
sim_uu = (R_user_norm @ R_user_norm.T).toarray().astype(np.float32)
#print(f"Sim(ux,ui) matrix: {sim_uu.shape}")

# Step 3.2: Keep Top-K Neighbors 
K = 40 #number that make precision at 10 to be the best(most)
np.fill_diagonal(sim_uu, 0) #set itself yo be a zero-->not be a neighbor
for i in range(NumUsers):
    row = sim_uu[i]
    if row.max() == 0: #skip
        continue
    top_idx = np.argsort(-row)[:K]
    mask = np.ones(NumUsers, dtype=bool); mask[top_idx] = False
    sim_uu[i, mask] = 0.0 #keep only top K neighbors, set the rest to zero


#Step 3.3: Predict Ratings — CFUB1 
#make intp matrix
R_dense = R_sparse.toarray().astype(np.float32) 

userbased_sumsim = np.abs(sim_uu).sum(axis=1, keepdims=True).clip(1) #|sum sim all|
userbased_matrix = (sim_uu @ R_dense) / userbased_sumsim  

userbased_norm = (userbased_matrix - userbased_matrix.min()) / (userbased_matrix.max() - userbased_matrix.min())

#clear memory 
del sim_uu, userbased_matrix
gc.collect() #delete obj that are not used anymore 


# Step 3.4: Evaluate
def userbased_score_fn(userId):
    row = user_to_idx.get(userId)
    return userbased_norm[row] if row is not None else np.zeros(NumMovies, dtype=np.float32)

p10_userbased = precision_at_10(userbased_score_fn, val_user_movie,train_user_watched, all_movies, movie_to_idx)
print(f"Model1:User-Based CF --> P@10 = {p10_userbased:.4f}")


#[4]model2 item based collaborative filtering 
print("==========[4]model2 item-based collaborative filtering==========")

#Step 4.1: Item–Item Cosine Similarity
R_item_norm = normalize(R_sparse.T, norm='l2')
sim_ii = (R_item_norm @ R_item_norm.T).toarray().astype(np.float32)
#print(f"Sim(Iy,Ij) matrix: {sim_ii.shape}")

#Step 4.2: Keep Top-K Similar Items
K = 40 #number that make precision at 10 to be the best(most)
np.fill_diagonal(sim_ii, 0)
for j in range(NumMovies):
    row = sim_ii[j]
    if row.max() == 0:
        continue
    top_idx = np.argsort(-row)[:K]
    mask = np.ones(NumMovies, dtype=bool); mask[top_idx] = False
    sim_ii[j, mask] = 0.0

#Step 4.3: Predict Ratings — CFIB1 
# r_hat(ux, Iy) = [sum_j sim(Iy,Ij)*r(ux,Ij)] / [sum |sim(Iy,Ij)|]

itembased_denom = np.abs(sim_ii).sum(axis=1, keepdims=True).T.clip(1)  
itembased_matrix = R_dense @ sim_ii.T / itembased_denom                        

itembased_norm = (itembased_matrix - itembased_matrix.min()) / (itembased_matrix.max() - itembased_matrix.min())

#clear memory
del sim_ii, itembased_matrix
gc.collect()

#Step 4.4: Evaluate 
def itembased_score_fn(userId):
    row = user_to_idx.get(userId)
    return itembased_norm[row] if row is not None else np.zeros(NumMovies, dtype=np.float32)

p10_ib = precision_at_10(itembased_score_fn, val_user_movie,train_user_watched, all_movies, movie_to_idx)
print(f"Model2:Item-Based CF --> P@10 = {p10_ib:.4f}")


#[5]model3 SVD
print("==========[5]model3 SVD==========")

#Step 5.1: Train SVD 
reader   = Reader(rating_scale=(1, 5))
sur_data = Dataset.load_from_df(train[["user_id","movie_id","rating"]], reader)
trainset = sur_data.build_full_trainset()

RANDOM_STATE = 42 #famous rendom number 555
svd = SVD(
    n_factors    = 50,    # l : a famous latent factors
    n_epochs     = 20,    # amount ofSGD iterations
    lr_all       = 0.005, # η : learning rate
    reg_all      = 0.02,  # λ : regularization
    random_state = RANDOM_STATE
)
svd.fit(trainset)

#Step 5.2: Compute r̂ = μ + bu + bi + P@Q^T 
mu = svd.trainset.global_mean #avg of all ratings in training set
movie_inner_ids, movie_raw_ids = [], []
for m in all_movies:
    try:
        movie_inner_ids.append(svd.trainset.to_inner_iid(m))
        movie_raw_ids.append(m)
    except ValueError: pass

qitem_sub = svd.qi[movie_inner_ids].astype(np.float32)
bitem_sub = svd.bi[movie_inner_ids].astype(np.float32)
pusers     = svd.pu.astype(np.float32)
busers     = svd.bu.astype(np.float32)

# r̂ = μ + bu + bi + P @ Q^T
svd_raw = pusers @ qitem_sub.T + busers[:, None] + bitem_sub[None, :] + mu  

svd_full = np.full((svd.trainset.n_users,NumMovies), mu, dtype=np.float32)
for j, m in enumerate(movie_raw_ids):
    svd_full[:, movie_to_idx[m]] = svd_raw[:, j]

svd_norm = (svd_full - svd_full.min()) / (svd_full.max() - svd_full.min())
svd_user_list  = [svd.trainset.to_raw_uid(i) for i in range(svd.trainset.n_users)]
user_to_svd_row = {uid: i for i, uid in enumerate(svd_user_list)}

#clear memory
del svd_full, svd_raw, qitem_sub, bitem_sub, pusers, busers
gc.collect()

#Step 5.3: Evaluate SVD 
def svd_score_fn(uid):
    row = user_to_svd_row.get(uid)
    return svd_norm[row] if row is not None else np.zeros(NumMovies, dtype=np.float32)

p10_svd = precision_at_10(svd_score_fn, val_user_movie,train_user_watched, all_movies, movie_to_idx)
print(f"Model3:SVD --> P@10 = {p10_svd:.4f}")
print("")



#[6]ensemble model
print("==========[6]ensemble model==========")
#Step 6.1: Pre-fetch Val User Score Rows
val_users = list(val_user_movie.keys()) #user from validation set
NumVal= len(val_users)

svd_rows = np.array([user_to_svd_row.get(uid, -1) for uid in val_users])
ub_rows  = np.array([user_to_idx.get(uid, -1)     for uid in val_users])
rel_idx  = np.array([movie_to_idx.get(val_user_movie[uid], -1) for uid in val_users])

# seen mask
seen_mask = np.zeros((NumVal, NumMovies), dtype=bool)
for ri, uid in enumerate(val_users):
    for m in train_user_watched.get(uid, set()):
        idx = movie_to_idx.get(m)
        if idx is not None:
            seen_mask[ri, idx] = True

fb = np.zeros(NumMovies, dtype=np.float32)
ub_v  = np.where(ub_rows[:,None]>=0,  userbased_norm[np.where(ub_rows>=0,  ub_rows,  0)], fb)
ib_v  = np.where(ub_rows[:,None]>=0,  itembased_norm[np.where(ub_rows>=0,  ub_rows,  0)], fb)
svd_v = np.where(svd_rows[:,None]>=0, svd_norm[np.where(svd_rows>=0,svd_rows, 0)], fb)

#Step 6.2: Bayesian Optimization for ensemble weights
print("Bayesian Optimization for ensemble weights (Optuna)")

import optuna

def objective(trial):
    a = trial.suggest_float("a", 0.1, 1.0)
    b = trial.suggest_float("b", 0.1, 1.0)
    c = trial.suggest_float("c", 0.1, 1.0)
    
    s = a + b + c
    w1, w2, w3 = a/s, b/s, c/s   # normalize

    # calculate ensemble score
    sc = w1*ub_v + w2*ib_v + w3*svd_v
    sc[seen_mask] = -np.inf

    top10 = np.argsort(sc, axis=1)[:, -10:]
    hits = sum(rel_idx[i] in top10[i] for i in range(NumVal) if rel_idx[i] >= 0)

    return hits / NumVal


# run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)  


# best result
best = study.best_trial

a = best.params["a"]
b = best.params["b"]
c = best.params["c"]
s = a + b + c

best_w = (a/s, b/s, c/s)
best_p10 = best.value

print(f"\nBest: UB={best_w[0]:.4f}  IB={best_w[1]:.4f}  SVD={best_w[2]:.4f}  →  P@10={best_p10:.4f}")

#Step 6.3: Final Ensemble 
W_UB, W_IB, W_SVD = best_w
print(f"Final weights: UB={W_UB:.1f}  IB={W_IB:.1f}  SVD={W_SVD:.1f}")

def ensemble_score_fn(userId):
    ub_row  = user_to_idx.get(userId)
    svd_row = user_to_svd_row.get(userId)
    s_ub  = userbased_norm[ub_row]   if ub_row  is not None else fb
    s_ib  = itembased_norm[ub_row]   if ub_row  is not None else fb
    s_svd = svd_norm[svd_row] if svd_row is not None else fb
    return W_UB * s_ub + W_IB * s_ib + W_SVD * s_svd

p10_ensemble = precision_at_10(ensemble_score_fn, val_user_movie,train_user_watched, all_movies, movie_to_idx)
print(f"\nEnsemble --> P@10 = {p10_ensemble:.4f}")




#==================================
print("\n==========[7]precision comparison==========")
movie_titles = movies.set_index("movie_id")["title"]

recommendations = {}

for uid in val_user_movie.keys():
    scores = ensemble_score_fn(uid).copy().astype(float)
    for m in train_user_watched.get(uid, set()):
        idx = movie_to_idx.get(m)
        if idx is not None: scores[idx] = -np.inf
    top10_idx   = np.argpartition(scores, -10)[-10:]
    top10_idx   = top10_idx[np.argsort(scores[top10_idx])[::-1]]
    recommendations[uid] = [all_movies[i] for i in top10_idx]

submission = pd.DataFrame([
    {"user_id": uid, "recommended_movies": ",".join(recs)}
    for uid, recs in recommendations.items()
])
submission.to_csv("recommendations.csv", index=False)
print(f"Saved recommendations.csv ")
submission.head(5)

# Sample Recommendation 
print("\nSample Recommendation:")
sample_uid = list(recommendations.keys())[0]
recs       = recommendations[sample_uid]
print(f"User {sample_uid} — Top-10 Recommendations:")
for rank, mid in enumerate(recs, 1):
    print(f"  {rank:2d}. {movie_titles.get(mid, mid):<45}  (id={mid})")
print(f"  → Relevant: {movie_titles.get(val_user_movie[sample_uid], val_user_movie[sample_uid])}")
