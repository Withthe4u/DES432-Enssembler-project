import pandas as pd
import numpy as np
from collections import defaultdict
import time
import warnings
warnings.filterwarnings('ignore')

# Surprise library
from surprise import Dataset, Reader, KNNBasic, KNNWithMeans, SVD
from surprise.model_selection import cross_validate
from surprise import accuracy

print('All imports successful!')


# Load training and validation sets
train_df = pd.read_csv('train.csv')
val_df   = pd.read_csv('val.csv')

# Load movies metadata (for display)
movies_df = pd.read_csv(
    'movies.dat', sep='::', engine='python',
    names=['movie_id', 'title', 'genres'],
    encoding='latin-1'
)

print(f'Train size : {len(train_df):,} ratings')
print(f'Val size   : {len(val_df):,} ratings')
print(f'Train users: {train_df.user_id.nunique():,}')
print(f'Val users  : {val_df.user_id.nunique():,}')
print(f'Movies     : {movies_df.movie_id.nunique():,}')
train_df.head()


import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Rating distribution
train_df['rating'].value_counts().sort_index().plot(kind='bar', ax=axes[0], color='steelblue')
axes[0].set_title('Rating Distribution')
axes[0].set_xlabel('Rating')
axes[0].set_ylabel('Count')

# Ratings per user
user_counts = train_df.groupby('user_id').size()
user_counts.hist(bins=50, ax=axes[1], color='salmon')
axes[1].set_title('Ratings per User')
axes[1].set_xlabel('# Ratings')

# Ratings per movie
movie_counts = train_df.groupby('movie_id').size()
movie_counts.hist(bins=50, ax=axes[2], color='mediumseagreen')
axes[2].set_title('Ratings per Movie')
axes[2].set_xlabel('# Ratings')

plt.tight_layout()
plt.savefig('eda.png', dpi=100)
plt.show()

print(f'Avg ratings/user  : {user_counts.mean():.1f}')
print(f'Avg ratings/movie : {movie_counts.mean():.1f}')
print(f'Sparsity          : {1 - len(train_df)/(train_df.user_id.nunique()*train_df.movie_id.nunique()):.4f}')


# Build Surprise Dataset from train_df
reader = Reader(rating_scale=(1, 5))
train_data = Dataset.load_from_df(train_df[['user_id', 'movie_id', 'rating']], reader)
trainset = train_data.build_full_trainset()  # Use ALL training data

# Precompute: movies each user has already rated (to exclude from recommendations)
user_rated = train_df.groupby('user_id')['movie_id'].apply(set).to_dict()

# All movie IDs in the training set
all_movies = set(train_df['movie_id'].unique())

# Validation: ground truth relevant movie per user (rating > 3)
val_relevant = val_df[val_df['rating'] > 3].groupby('user_id')['movie_id'].apply(set).to_dict()
val_users = val_df['user_id'].unique()

print(f'Trainset: {trainset.n_ratings:,} ratings, {trainset.n_users} users, {trainset.n_items} items')
print(f'Val users: {len(val_users)}')



def get_top10_recommendations(algo, user_id, user_rated, all_movies, n=10):
    """
    For a given user, predict scores for all unseen movies and return top-N.
    
    Speed tips applied:
    - Only predict on movies NOT already rated by the user
    - Use list comprehension (faster than loops)
    - Use np.argpartition instead of full sort (O(n) vs O(n log n))
    """
    rated = user_rated.get(user_id, set())
    candidates = list(all_movies - rated)
    
    # Batch predict
    preds = [algo.predict(user_id, mid).est for mid in candidates]
    
    # Top-N via argpartition (much faster than sorted for large candidate sets)
    if len(candidates) <= n:
        return candidates
    top_idx = np.argpartition(preds, -n)[-n:]
    return [candidates[i] for i in top_idx]


def compute_precision_at_10(recommendations_dict, val_relevant):
    """
    Compute overall Precision@10 across all validation users.
    val_relevant: {user_id: set of relevant movie_ids}
    recommendations_dict: {user_id: list of 10 recommended movie_ids}
    """
    scores = []
    for uid, recs in recommendations_dict.items():
        relevant = val_relevant.get(uid, set())
        hits = len(set(recs) & relevant)
        scores.append(hits / 10)
    return np.mean(scores)


def generate_all_recommendations(algo, val_users, user_rated, all_movies):
    """
    Generate top-10 recommendations for all validation users.
    Returns dict: {user_id: [movie_id, ...]}
    Also returns raw score dict: {user_id: {movie_id: score}} for ensemble.
    """
    recs = {}
    scores = {}
    for uid in val_users:
        rated = user_rated.get(uid, set())
        candidates = list(all_movies - rated)
        preds = {mid: algo.predict(uid, mid).est for mid in candidates}
        scores[uid] = preds
        top10 = sorted(preds, key=preds.get, reverse=True)[:10]
        recs[uid] = top10
    return recs, scores


print('Helper functions defined.')



print('Training User-Based CF...')
t0 = time.time()

ubcf = KNNWithMeans(
    k=40,            # top-40 similar users
    min_k=5,         # need at least 5 neighbors
    sim_options={
        'name': 'pearson_baseline',  # best for CF
        'user_based': True,          # user-user CF
        'min_support': 3,            # min co-rated items
    },
    verbose=False
)
ubcf.fit(trainset)
print(f'User-Based CF trained in {time.time()-t0:.1f}s')

# Generate recommendations
print('Generating recommendations for validation users...')
t0 = time.time()
ubcf_recs, ubcf_scores = generate_all_recommendations(ubcf, val_users, user_rated, all_movies)
print(f'Done in {time.time()-t0:.1f}s')

ubcf_p10 = compute_precision_at_10(ubcf_recs, val_relevant)
print(f'\n>>> User-Based CF Precision@10 = {ubcf_p10:.4f}')



print('Training Item-Based CF...')
t0 = time.time()

ibcf = KNNWithMeans(
    k=40,
    min_k=3,
    sim_options={
        'name': 'pearson_baseline',
        'user_based': False,   # item-item CF
        'min_support': 3,
    },
    verbose=False
)
ibcf.fit(trainset)
print(f'Item-Based CF trained in {time.time()-t0:.1f}s')

print('Generating recommendations...')
t0 = time.time()
ibcf_recs, ibcf_scores = generate_all_recommendations(ibcf, val_users, user_rated, all_movies)
print(f'Done in {time.time()-t0:.1f}s')

ibcf_p10 = compute_precision_at_10(ibcf_recs, val_relevant)
print(f'\n>>> Item-Based CF Precision@10 = {ibcf_p10:.4f}')


print('Training SVD...')
t0 = time.time()

svd = SVD(
    n_factors=100,   # latent dimensions
    n_epochs=30,     # SGD iterations (30 is enough, 20 is faster)
    lr_all=0.005,    # learning rate
    reg_all=0.02,    # regularization
    biased=True,     # include user/item bias terms
    random_state=42,
    verbose=False
)
svd.fit(trainset)
print(f'SVD trained in {time.time()-t0:.1f}s')

print('Generating recommendations...')
t0 = time.time()
svd_recs, svd_scores = generate_all_recommendations(svd, val_users, user_rated, all_movies)
print(f'Done in {time.time()-t0:.1f}s')

svd_p10 = compute_precision_at_10(svd_recs, val_relevant)
print(f'\n>>> SVD Precision@10 = {svd_p10:.4f}')

results = {
    'User-Based CF': ubcf_p10,
    'Item-Based CF': ibcf_p10,
    'SVD':           svd_p10,
}

print('='*40)
print(f'{"Model":<20} {"Precision@10"}')
print('-'*40)
for name, score in results.items():
    print(f'{name:<20} {score:.4f}')
print('='*40)


# Min-max normalize scores per user so models are on the same scale
def normalize_scores(scores_dict):
    """Normalize each user's scores to [0, 1]"""
    norm = {}
    for uid, movie_scores in scores_dict.items():
        vals = np.array(list(movie_scores.values()))
        mn, mx = vals.min(), vals.max()
        rng = mx - mn if mx > mn else 1.0
        norm[uid] = {mid: (s - mn) / rng for mid, s in movie_scores.items()}
    return norm

ubcf_norm = normalize_scores(ubcf_scores)
ibcf_norm = normalize_scores(ibcf_scores)
svd_norm  = normalize_scores(svd_scores)

print('Scores normalized.')


def ensemble_recommendations(ubcf_norm, ibcf_norm, svd_norm,
                             val_users, user_rated, all_movies,
                             w1, w2, w3):
    """
    Combine three models with weights w1, w2, w3.
    Returns recommendation dict {user_id: [top-10 movie_ids]}
    """
    recs = {}
    for uid in val_users:
        rated = user_rated.get(uid, set())
        candidates = list(all_movies - rated)
        
        u_scores = ubcf_norm.get(uid, {})
        i_scores = ibcf_norm.get(uid, {})
        s_scores = svd_norm.get(uid, {})
        
        combined = {
            mid: (w1 * u_scores.get(mid, 0.0)
                + w2 * i_scores.get(mid, 0.0)
                + w3 * s_scores.get(mid, 0.0))
            for mid in candidates
        }
        recs[uid] = sorted(combined, key=combined.get, reverse=True)[:10]
    return recs


# ---- Grid Search ----
print('Running grid search for ensemble weights...')
step = 0.05
weights = np.arange(0, 1 + step, step)

best_score = -1
best_weights = (0, 0, 1)
grid_results = []

t0 = time.time()
for w1 in weights:
    for w2 in weights:
        w3 = round(1 - w1 - w2, 4)
        if w3 < 0 or w3 > 1:
            continue
        recs = ensemble_recommendations(
            ubcf_norm, ibcf_norm, svd_norm,
            val_users, user_rated, all_movies,
            w1, w2, w3
        )
        p10 = compute_precision_at_10(recs, val_relevant)
        grid_results.append((w1, w2, w3, p10))
        if p10 > best_score:
            best_score = p10
            best_weights = (w1, w2, w3)

print(f'Grid search done in {time.time()-t0:.1f}s ({len(grid_results)} combinations)')
print(f'\n>>> Best weights: UBCF={best_weights[0]:.2f}, IBCF={best_weights[1]:.2f}, SVD={best_weights[2]:.2f}')
print(f'>>> Best Ensemble Precision@10 = {best_score:.4f}')



# Visualize the grid search results
grid_df = pd.DataFrame(grid_results, columns=['w_ubcf', 'w_ibcf', 'w_svd', 'precision'])

# Pivot for heatmap (w_svd fixed at best, vary w1 and w2)
pivot = grid_df.pivot_table(values='precision', index='w_ubcf', columns='w_ibcf', aggfunc='max')

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.figure(figsize=(9, 7))
plt.title('Grid Search: Precision@10 by Weight Combination\n(max over all w_svd values)', fontsize=13)
im = plt.imshow(pivot, aspect='auto', cmap='YlOrRd', origin='lower')
plt.colorbar(im, label='Precision@10')
plt.xlabel('w_ibcf')
plt.ylabel('w_ubcf')
plt.xticks(range(len(pivot.columns)), [f'{v:.2f}' for v in pivot.columns], rotation=90)
plt.yticks(range(len(pivot.index)), [f'{v:.2f}' for v in pivot.index])
plt.tight_layout()
plt.savefig('grid_search_heatmap.png', dpi=100)
plt.show()


# Generate final ensemble recommendations with best weights
final_recs = ensemble_recommendations(
    ubcf_norm, ibcf_norm, svd_norm,
    val_users, user_rated, all_movies,
    *best_weights
)

# Summary table
print('='*50)
print(f'{"Model":<25} {"Precision@10"}')
print('-'*50)
print(f'{"User-Based CF":<25} {ubcf_p10:.4f}')
print(f'{"Item-Based CF":<25} {ibcf_p10:.4f}')
print(f'{"SVD":<25} {svd_p10:.4f}')
print('-'*50)
print(f'{"Ensemble (best weights)":<25} {best_score:.4f}')
print('='*50)
print(f'\nBest weights → UBCF: {best_weights[0]:.2f}, IBCF: {best_weights[1]:.2f}, SVD: {best_weights[2]:.2f}')

improvement = best_score - max(ubcf_p10, ibcf_p10, svd_p10)
print(f'Ensemble improvement over best single model: {improvement:+.4f}')