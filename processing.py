import pandas as pd

def cleanData():
    
    # load all files(MOVIES.DAT HAVE NOT BEEN USED YET. WILL NEED WHEN WE WANT TO USE ONE HOT ENCODED)
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

    #Drop empty and duplicates(training set only) dropping duplicates on val is not necessary
    train = train.drop(columns=["timestamp"]).dropna().drop_duplicates()
    val = val.drop(columns=["timestamp"]).dropna()

    #Convert all IDs into string so model cannot mistake it for number
    train["user_id"] = train["user_id"].astype(str)
    train["movie_id"] = train["movie_id"].astype(str)

    val["user_id"] = val["user_id"].astype(str)
    val["movie_id"] = val["movie_id"].astype(str)

    movies["movie_id"] = movies["movie_id"].astype(str)
    users["user_id"] = users["user_id"].astype(str)

    return train, val, movies, users

def groupData(train, val):
    #Grouping all movies that each user already watched into a dict eg. "1": {"1","2","3"} --> user 1 already seen movie 1,2,3
    #note that dict is not sorted numerically bc it is a string eg. {"1", "10", "100"}
    train_user_watched = train.groupby("user_id")["movie_id"].apply(set).to_dict()
    
    #creating dict for validation eg. "1":{"10"} --> if our model recommend movie 10 for user 1 it is "correct"
    val_user_movie = val.set_index("user_id")["movie_id"].to_dict()
    
    #all of the movies
    all_movies = train["movie_id"].unique()

    return train_user_watched, val_user_movie, all_movies

#Function to get all movies that the user has not seen yet
def get_candidate_movies(user_id, train_user_watched, all_movies):
    seen = train_user_watched[user_id]
    return [m for m in all_movies if m not in seen]

#Function to get all processed data
def allData():
    train, val, movies, users = cleanData()

    train_user_watched, val_user_movie, all_movies = groupData(train, val)

    return {
        "train": train,  #cleaned training data
        "val": val,  #validation data
        "movies": movies,  #unfinished movies data
        "users": users,  #cleaned user.dat
        "train_user_watched": train_user_watched,  #dict of movies the user has already watched
        "val_user_movie": val_user_movie,  # dict for precision calculation
        "all_movies": all_movies #list of all unique movies from training set
    }
    
