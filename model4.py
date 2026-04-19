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
train = data["train"]
val= data["val"]
movies= data["movies"]
print(len(data["train"]))
print(len(data["val"])) 


#construct utility matrix
all_movies = list(train["movie_id"].unique())
all_users  = list(train["user_id"].unique())
movie_to_idx = {m: i for i, m in enumerate(all_movies)}
user_to_idx  = {u: i for i, u in enumerate(all_users)}
n_users, n_movies = len(all_users), len(all_movies)
# Sparse rating matrix R — shape (n_users, n_movies)
rows = train["user_id"].map(user_to_idx).values
cols = train["movie_id"].map(movie_to_idx).values
vals = train["rating"].values.astype(np.float32)
R_sparse = csr_matrix((vals, (rows, cols)), shape=(n_users, n_movies))
# Helper dicts
train_user_watched = train.groupby("user_id")["movie_id"].apply(set).to_dict()
val_user_movie     = val.set_index("user_id")["movie_id"].to_dict()
sparsity = 1 - R_sparse.nnz / (n_users * n_movies)
print(f"Utility matrix : {R_sparse.shape}  ({R_sparse.nnz:,} observed ratings)")
print(f"Sparsity       : {sparsity:.4f}  ({sparsity*100:.1f}% empty)")
print(f"Val users      : {len(val_user_movie)}")


#precision calculation
def precision_at_10(score_fn, val_user_movie, train_user_watched, all_movies, movie_to_idx):
    hits = 0
    for uid, relevant in val_user_movie.items():
        scores = score_fn(uid).copy().astype(float)
        for m in train_user_watched.get(uid, set()):
            idx = movie_to_idx.get(m)
            if idx is not None: scores[idx] = -np.inf
        top10 = [all_movies[i] for i in np.argpartition(scores, -10)[-10:]]
        if relevant in top10: hits += 1
    return hits / len(val_user_movie)

print(f"Random baseline P@10 ≈ {10/n_movies:.4f}")
print("Evaluator ready.")







#model1 user based CF
print("Model1: User-Based CF ====================================   ")
# ── Step 3.1: User–User Cosine Similarity ────────────────────────────────────
# normalize แต่ละ row ของ R → dot product = cosine similarity
t0 = time.time()
R_user_norm = normalize(R_sparse, norm='l2')
sim_uu = (R_user_norm @ R_user_norm.T).toarray().astype(np.float32)
print(f"User–user sim matrix: {sim_uu.shape}  ({time.time()-t0:.1f}s)")
print(f"Sim range: [{sim_uu[sim_uu>0].min():.3f}, {sim_uu.max():.3f}]")
# ── Step 3.2: Keep Top-K Neighbors ───────────────────────────────────────────
K = 40
t0 = time.time()
np.fill_diagonal(sim_uu, 0)   # ไม่นับตัวเอง
for i in range(n_users):
    row = sim_uu[i]
    if row.max() == 0: continue
    top_idx = np.argpartition(row, -K)[-K:]
    mask = np.ones(n_users, dtype=bool); mask[top_idx] = False
    sim_uu[i, mask] = 0.0
print(f"Top-{K} filter done ({time.time()-t0:.1f}s)  avg neighbors={( sim_uu>0).sum(axis=1).mean():.1f}")
# ── Step 3.3: Predict Ratings — CFUB1 Formula ────────────────────────────────
# numerator   = sim_uu  @ R_dense    (weighted sum of ratings)
# denominator = sum |sim| per user

t0 = time.time()
R_dense = R_sparse.toarray().astype(np.float32)

ub_denom = np.abs(sim_uu).sum(axis=1, keepdims=True).clip(1)
ub_matrix = (sim_uu @ R_dense) / ub_denom                         # (n_users, n_movies)

ub_norm = (ub_matrix - ub_matrix.min()) / (ub_matrix.max() - ub_matrix.min())
print(f"UB score matrix: {ub_matrix.shape}  ({time.time()-t0:.1f}s)")

# ── Free sim_uu (ไม่ต้องใช้แล้ว) ─────────────────────────────────────────────
del sim_uu, ub_matrix; gc.collect()
print("sim_uu freed.")
# ── Step 3.4: Evaluate ───────────────────────────────────────────────────────
def ub_score_fn(uid):
    row = user_to_idx.get(uid)
    return ub_norm[row] if row is not None else np.zeros(n_movies, dtype=np.float32)

t0 = time.time()
p10_ub = precision_at_10(ub_score_fn, val_user_movie,
                          train_user_watched, all_movies, movie_to_idx)
print(f"Model 1 — User-Based CF  |  P@10 = {p10_ub:.4f}  ({time.time()-t0:.1f}s)")


#model2 item based CF
print("Model2 Item-Based CF ====================================   ")
# ── Step 4.1: Item–Item Cosine Similarity ────────────────────────────────────
t0 = time.time()
R_item_norm = normalize(R_sparse.T, norm='l2')
sim_ii = (R_item_norm @ R_item_norm.T).toarray().astype(np.float32)
print(f"Item–item sim matrix: {sim_ii.shape}  ({time.time()-t0:.1f}s)")
# ── Step 4.2: Keep Top-K Similar Items ───────────────────────────────────────
t0 = time.time()
np.fill_diagonal(sim_ii, 0)
for j in range(n_movies):
    row = sim_ii[j]
    if row.max() == 0: continue
    top_idx = np.argpartition(row, -K)[-K:]
    mask = np.ones(n_movies, dtype=bool); mask[top_idx] = False
    sim_ii[j, mask] = 0.0
print(f"Top-{K} filter done ({time.time()-t0:.1f}s)")
# ── Step 4.3: Predict Ratings — CFIB1 Formula ────────────────────────────────
# r_hat(ux, Iy) = [sum_j sim(Iy,Ij)*r(ux,Ij)] / [sum |sim(Iy,Ij)|]
# vectorized: R_dense @ sim_ii.T / denom

t0 = time.time()
ib_denom = np.abs(sim_ii).sum(axis=1, keepdims=True).T.clip(1)   # (1, n_movies)
ib_matrix = R_dense @ sim_ii.T / ib_denom                         # (n_users, n_movies)

ib_norm = (ib_matrix - ib_matrix.min()) / (ib_matrix.max() - ib_matrix.min())
print(f"IB score matrix: {ib_matrix.shape}  ({time.time()-t0:.1f}s)")

del sim_ii, ib_matrix; gc.collect()
print("sim_ii freed.")
# ── Step 4.4: Show Similar Movies Example ────────────────────────────────────
# (ตัวอย่าง top-5 movies ที่ similar กับ Toy Story จาก ib_norm)
# Note: เนื่องจาก sim_ii ถูก del แล้ว จึงใช้ ib_norm score สำหรับ user ที่ rate Toy Story

target_mid = "1"   # Toy Story
movie_titles = movies.set_index("movie_id")["title"]
if target_mid in movie_to_idx:
    # user ที่ rate Toy Story สูง: ดู ib scores column ของ Toy Story
    toy_col = ib_norm[:, movie_to_idx[target_mid]]
    # top users that rated Toy Story highly
    top_u_idx = np.argpartition(toy_col, -3)[-3:]
    for ui in top_u_idx:
        uid = all_users[ui]
        score = toy_col[ui]
        print(f"User {uid}: predicted IB score for Toy Story = {score:.3f}")

print(f"\nItem-Based model ready. Movies indexed: {n_movies}")
# ── Step 4.5: Evaluate ───────────────────────────────────────────────────────
def ib_score_fn(uid):
    row = user_to_idx.get(uid)
    return ib_norm[row] if row is not None else np.zeros(n_movies, dtype=np.float32)

t0 = time.time()
p10_ib = precision_at_10(ib_score_fn, val_user_movie,
                          train_user_watched, all_movies, movie_to_idx)
print(f"Model 2 — Item-Based CF  |  P@10 = {p10_ib:.4f}  ({time.time()-t0:.1f}s)")

#model3 SVD
print("Model3 SVD ====================================   ")
# ── Step 5.1: Train SVD ───────────────────────────────────────────────────────
t0 = time.time()
reader   = Reader(rating_scale=(1, 5))
sur_data = Dataset.load_from_df(train[["user_id","movie_id","rating"]], reader)
trainset = sur_data.build_full_trainset()

RANDOM_STATE = 42
svd = SVD(
    n_factors    = 50,    # l : จำนวน latent factors
    n_epochs     = 20,    # จำนวน SGD iterations
    lr_all       = 0.005, # η : learning rate
    reg_all      = 0.02,  # λ : regularization
    random_state = RANDOM_STATE
)
svd.fit(trainset)
print(f"SVD training : {time.time()-t0:.1f}s")
print(f"l (factors)  : {svd.n_factors}")
print(f"μ (global mean) : {svd.trainset.global_mean:.3f}")
print(f"P shape (n_users × l) : {svd.pu.shape}")
print(f"Q shape (n_items × l) : {svd.qi.shape}")

# ── Step 5.2: Compute r̂ = μ + bu + bi + P@Q^T (Vectorized) ──────────────────
# ตรงกับ Slide 9.2.3 eq. 9.7  + bias terms จาก Slide 9.3
t0 = time.time()
mu = svd.trainset.global_mean

movie_inner_ids, movie_raw_ids = [], []
for m in all_movies:
    try:
        movie_inner_ids.append(svd.trainset.to_inner_iid(m))
        movie_raw_ids.append(m)
    except ValueError: pass

qi_sub = svd.qi[movie_inner_ids].astype(np.float32)
bi_sub = svd.bi[movie_inner_ids].astype(np.float32)
pu     = svd.pu.astype(np.float32)
bu     = svd.bu.astype(np.float32)

# r̂ = μ + bu + bi + P @ Q^T
svd_raw = pu @ qi_sub.T + bu[:, None] + bi_sub[None, :] + mu   # (n_svd_users, n_known_movies)

svd_full = np.full((svd.trainset.n_users, n_movies), mu, dtype=np.float32)
for j, m in enumerate(movie_raw_ids):
    svd_full[:, movie_to_idx[m]] = svd_raw[:, j]

svd_norm = (svd_full - svd_full.min()) / (svd_full.max() - svd_full.min())
svd_user_list  = [svd.trainset.to_raw_uid(i) for i in range(svd.trainset.n_users)]
user_to_svd_row = {uid: i for i, uid in enumerate(svd_user_list)}

del svd_full, svd_raw, qi_sub, bi_sub, pu, bu; gc.collect()
print(f"SVD score matrix done ({time.time()-t0:.2f}s)  shape={svd_norm.shape}")

# ── Step 5.3: Visualise Latent Factors (PCA) ──────────────────────────────────
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=RANDOM_STATE)
qi_2d = pca.fit_transform(svd.qi)

popular_mids = train.groupby("movie_id")["rating"].count()
popular_mids = popular_mids[popular_mids >= 200].index.tolist()

movie_titles = movies.set_index("movie_id")["title"]
coords = []
for m in popular_mids[:400]:
    try:
        ii = svd.trainset.to_inner_iid(m)
        coords.append(qi_2d[ii])
    except: pass
coords = np.array(coords)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(coords[:,0], coords[:,1], alpha=0.35, s=12, c="#4C72B0")
ax.set_title(f"Item Latent Factors — PCA Projection (l={svd.n_factors})", fontsize=13)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
plt.tight_layout(); plt.savefig("svd_pca.png", dpi=150, bbox_inches='tight'); plt.show()
# ── Step 5.4: Evaluate SVD ────────────────────────────────────────────────────
def svd_score_fn(uid):
    row = user_to_svd_row.get(uid)
    return svd_norm[row] if row is not None else np.zeros(n_movies, dtype=np.float32)

t0 = time.time()
p10_svd = precision_at_10(svd_score_fn, val_user_movie,
                           train_user_watched, all_movies, movie_to_idx)
print(f"Model 3 — SVD  |  P@10 = {p10_svd:.4f}  ({time.time()-t0:.1f}s)")

#ensemble model
# ── Step 6.1: Pre-fetch Val User Score Rows ──────────────────────────────────
val_users = list(val_user_movie.keys())
n_val     = len(val_users)

svd_rows = np.array([user_to_svd_row.get(uid, -1) for uid in val_users])
ub_rows  = np.array([user_to_idx.get(uid, -1)     for uid in val_users])
rel_idx  = np.array([movie_to_idx.get(val_user_movie[uid], -1) for uid in val_users])

# seen mask
seen_mask = np.zeros((n_val, n_movies), dtype=bool)
for ri, uid in enumerate(val_users):
    for m in train_user_watched.get(uid, set()):
        idx = movie_to_idx.get(m)
        if idx is not None: seen_mask[ri, idx] = True

fb = np.zeros(n_movies, dtype=np.float32)
ub_v  = np.where(ub_rows[:,None]>=0,  ub_norm[np.where(ub_rows>=0,  ub_rows,  0)], fb)
ib_v  = np.where(ub_rows[:,None]>=0,  ib_norm[np.where(ub_rows>=0,  ub_rows,  0)], fb)
svd_v = np.where(svd_rows[:,None]>=0, svd_norm[np.where(svd_rows>=0,svd_rows, 0)], fb)

print(f"Pre-fetched: ub={ub_v.shape}  ib={ib_v.shape}  svd={svd_v.shape}")
# ── Step 6.2: Grid Search ────────────────────────────────────────────────────
def evaluate_ensemble(w1, w2, w3):
    """P@10 ของ ensemble weights (w1=UB, w2=IB, w3=SVD)"""
    sc = w1*ub_v + w2*ib_v + w3*svd_v
    sc[seen_mask] = -np.inf
    top10 = np.argpartition(sc, -10, axis=1)[:, -10:]
    hits = sum(rel_idx[i] in top10[i] for i in range(n_val) if rel_idx[i]>=0)
    return hits / n_val

t0 = time.time()
weight_grid  = np.arange(0, 1.01, 0.1).round(1)
best_p10, best_w = 0.0, (1/3, 1/3, 1/3)
grid_results = []

for w1 in weight_grid:
    for w2 in weight_grid:
        w3 = round(1.0 - w1 - w2, 1)
        if not (0.0 <= w3 <= 1.0): continue
        p = evaluate_ensemble(w1, w2, w3)
        grid_results.append({"w_ub":w1, "w_ib":w2, "w_svd":w3, "p10":p})
        if p > best_p10: best_p10=p; best_w=(w1,w2,w3)

grid_df = pd.DataFrame(grid_results).sort_values("p10", ascending=False)
print(f"Grid search: {len(grid_results)} combinations in {time.time()-t0:.1f}s")
print(f"\nBest: UB={best_w[0]:.1f}  IB={best_w[1]:.1f}  SVD={best_w[2]:.1f}  →  P@10={best_p10:.4f}")
print("\nTop-10 weight combinations:")
print(grid_df.head(10).to_string(index=False))
# ── Step 6.3: Visualise Grid Search ──────────────────────────────────────────
pivot = grid_df.pivot_table(index="w_ub", columns="w_ib", values="p10", aggfunc="max")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlGnBu",
            linewidths=0.4, cbar_kws={"label":"Precision@10"}, ax=ax)
ax.set_title("Ensemble Grid Search — Precision@10\n(w_svd = 1 − w_ub − w_ib)", fontsize=13)
ax.set_xlabel("w_ib"); ax.set_ylabel("w_ub")
plt.tight_layout(); plt.savefig("ensemble_grid.png", dpi=150, bbox_inches='tight'); plt.show()
# ── Step 6.4: Final Ensemble Score Function ──────────────────────────────────
W_UB, W_IB, W_SVD = best_w
print(f"Final weights: UB={W_UB:.1f}  IB={W_IB:.1f}  SVD={W_SVD:.1f}")

def ensemble_score_fn(uid):
    ub_row  = user_to_idx.get(uid)
    svd_row = user_to_svd_row.get(uid)
    s_ub  = ub_norm[ub_row]   if ub_row  is not None else fb
    s_ib  = ib_norm[ub_row]   if ub_row  is not None else fb
    s_svd = svd_norm[svd_row] if svd_row is not None else fb
    return W_UB * s_ub + W_IB * s_ib + W_SVD * s_svd

t0 = time.time()
p10_ensemble = precision_at_10(ensemble_score_fn, val_user_movie,
                                train_user_watched, all_movies, movie_to_idx)
print(f"\nEnsemble  |  P@10 = {p10_ensemble:.4f}  ({time.time()-t0:.1f}s)")


#Final
# ── Summary Table ─────────────────────────────────────────────────────────────
summary = pd.DataFrame({
    "Model"         : ["User-Based CF","Item-Based CF","SVD Latent Factor","Ensemble"],
    "Slide"         : ["8.3.1 CFUB1", "8.3.2 CFIB1", "9.2", "9.4"],
    "Precision@10"  : [p10_ub, p10_ib, p10_svd, p10_ensemble],
    "Details"       : [f"cosine sim, K={K}",
                       f"cosine sim, K={K}",
                       f"l={svd.n_factors}, {svd.n_epochs} epochs, SGD+reg",
                       f"UB={W_UB:.1f} IB={W_IB:.1f} SVD={W_SVD:.1f} (grid search)"]
})
disp = summary.copy()
disp["Precision@10"] = disp["Precision@10"].map("{:.4f}".format)
print(disp.to_string(index=False))

# ── Generate & Save recommendations.csv ──────────────────────────────────────
t0 = time.time()
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
print(f"Saved recommendations.csv  ({time.time()-t0:.2f}s)")
submission.head(5)

# ── Sample Recommendation ─────────────────────────────────────────────────────
sample_uid = list(recommendations.keys())[0]
recs       = recommendations[sample_uid]
print(f"User {sample_uid} — Top-10 Recommendations:")
for rank, mid in enumerate(recs, 1):
    print(f"  {rank:2d}. {movie_titles.get(mid, mid):<45}  (id={mid})")
print(f"  → Relevant: {movie_titles.get(val_user_movie[sample_uid], val_user_movie[sample_uid])}")
