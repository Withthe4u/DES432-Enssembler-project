from processing import allData, get_candidate_movies
import math


#import data
data = allData()
train = data["train"]

'''

#checking if the data is loaded correctly

print(len(data["train"]))
print(len(data["val"]))


#construct matrix
def create_matrix(train):
    return train.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating"
    )

matrix = create_matrix(data["train"])

#similarity function
def cosine_sim(v1, v2):
    dot = sum(a*b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum((a*a) for a in v1))
    norm2 = math.sqrt(sum((b*b) for b in v2))
    
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)

#item similarity
def build_item_similarity(matrix):
    matrix = matrix.fillna(0)
    movies = list(matrix.columns)
    
    item_sim = {}

    for i in range(len(movies)):
        m1 = movies[i]
        v1 = matrix[m1].values
        
        item_sim[m1] = {}
        
        for j in range(len(movies)):
            m2 = movies[j]
            v2 = matrix[m2].values
            
            sim = cosine_sim(v1, v2)
            item_sim[m1][m2] = sim

    return item_sim

#model 
def predict_item_cf(user_id, matrix, item_sim):
    if user_id not in matrix.index:
        return {}

    user_ratings = matrix.loc[user_id].fillna(0)

    scores = {}

    for target_movie in matrix.columns:
        total = 0
        
        for seen_movie, rating in user_ratings.items():
            if rating > 0:
                sim = item_sim[target_movie].get(seen_movie, 0)
                total += sim * rating
        
        scores[target_movie] = total

    return scores

#recommendation function
def recommend_item_cf(user_id, matrix, item_sim, data):
    scores = predict_item_cf(user_id, matrix, item_sim)

    seen = data["train_user_watched"].get(user_id, set())

    candidates = [
        (m, s) for m, s in scores.items() if m not in seen
    ]

    candidates.sort(key=lambda x: x[1], reverse=True)

    return [m for m, _ in candidates[:10]]

#evaluate
def precision_at_10(recs, actual):
    return 0.1 if actual in recs else 0


def evaluate_item_cf(matrix, item_sim, data):
    scores = []

    for user, actual in data["val_user_movie"].items():
        recs = recommend_item_cf(user, matrix, item_sim, data)
        scores.append(precision_at_10(recs, actual))

    return sum(scores) / len(scores)

matrix = create_matrix(data["train"])
item_sim = build_item_similarity(matrix)

score = evaluate_item_cf(matrix, item_sim, data)

print("Item-CF Precision@10:", score)

'''
