import pandas as pd

# load all files(MOVIES.DAT HAVE NOT BEEN USED YET)
train = pd.read_csv("train.csv")
val = pd.read_csv("val.csv")
movies = pd.read_csv(
    "movies.dat",
    sep="::",
    engine="python",
    names=["movie_id", "title", "genres"],
    encoding="latin-1"
)
users = pd.read_csv(
    "users.dat",
    sep="::",
    engine="python",
    names=["user_id", "gender", "age", "occupation", "zip"],
    encoding="latin-1"
)

#Drop "timestamp" because it is unnecessary
train = train.drop(columns=["timestamp"])
val = val.drop(columns=["timestamp"])

#Drop empty and duplicates(training set only) dropping duplicates on val is not necessary
train = train.dropna().drop_duplicates()
val = val.dropna()

#Convert all IDs into string so model cannot mistake it for number
train["user_id"] = train["user_id"].astype(str)
train["movie_id"] = train["movie_id"].astype(str)

val["user_id"] = val["user_id"].astype(str)
val["movie_id"] = val["movie_id"].astype(str)

movies["movie_id"] = movies["movie_id"].astype(str)
users["user_id"] = users["user_id"].astype(str)

#Grouping all movies that each user already watched into a dict eg. "1": {"1","2","3"} --> user 1 already seen movie 1,2,3
#note that dict is not sorted numerically bc it is a string eg. {"1", "10", "100"}
train_user_watched = train.groupby("user_id")["movie_id"].apply(set).to_dict()

#creating dict for movies that should be recommended for faster search later "1":{"10"} --> should recommended movie 10 to user 1
val_user_movie = val.set_index("user_id")["movie_id"].to_dict()

#all of the movies
all_movies = train["movie_id"].unique()

#Function to get all movies that the user has not seen yet
def get_candidate_movies(user_id):
    seen = train_user_watched[user_id]
    return [m for m in all_movies if m not in seen]
