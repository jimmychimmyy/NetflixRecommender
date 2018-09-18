import numpy as np
import pandas as pd
import os
import operator
import csv
import dask.dataframe as dd
import dask.multiprocessing
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing.dummy import Pool
import multiprocessing
import functools
import time
import warnings
from joblib import Parallel, delayed

FILEPATH_MOVIE_TITLES = "./netflix-prize-data/movie_titles.csv"
FILEPATH_COMBINED_DATA = "./netflix-prize-data/combined_data_*"
FILEPATH_COMBINED_DATA_1 = "./netflix-prize-data/combined_data_1.txt"
FILEPATH_COMBINED_DATA_2 = "./netflix-prize-data/combined_data_2.txt"
FILEPATH_COMBINED_DATA_3 = "./netflix-prize-data/combined_data_3.txt"
FILEPATH_COMBINED_DATA_4 = "./netflix-prize-data/combined_data_4.txt"
FILEPATH_PROBE = "./netflix-prize-data/probe.txt"
FILEPATH_QUALIFYING = "./netflix-prize-data/qualifying.txt"

CLEAN = "./netflix_data_clean/clean_data_*.csv"
CLEAN_1 = "./netflix_data_clean/clean_data_1.csv"
CLEAN_2 = "./netflix_data_clean/clean_data_2.csv"
CLEAN_3 = "./netflix_data_clean/clean_data_3.csv"
CLEAN_4 = "./netflix_data_clean/clean_data_4.csv"

class NetflixRecommender():

    def __init__(self):
        self.movie_id = "1"
        self.customer_id = "1488844"

    def read_train_data(self):
        # Read Rating data into df
        df = dd.read_csv(CLEAN)
        print("df.head()", '\n', df.head(10))
        return df

    def read_test_data(self):
        df_probe = dd.read_csv(FILEPATH_PROBE, sep='\n', header=None, names=['info']).compute()
        print("df_probe.head():", df_probe.head())
        return df_probe

    def get_movie_indices(self):
        movie_rows = self.df[(self.df['movie_id'].str.contains(':'))] #.compute()
        movie_indices = list(movie_rows.head(-1).index)
        #print("len(movie_indices):", len(movie_indices))
        #print("movie_indicies:", movie_indices)
        return movie_indices

    def get_num_movies(self):
        movies = set((self.df['movie_id'].head(-1).values).tolist())
        #print(movies)
        #print(len(movies))
        return len(movies)

    def get_feature_vector(self, user_id):
        my_movies = self.df.loc[self.df['customer_id'] == np.int64(user_id)]
        my_movies_list = (my_movies['movie_id'].head(-1).values).tolist()
        my_movies_ratings = (my_movies['rating'].head(-1).values).tolist()
        #print(my_movies_list)
        #print(my_movies_ratings)
        ratings = [0] * self.num_movies
        for movie, rating in zip(my_movies_list, my_movies_ratings):
            ratings[int(movie) - 1] = int(rating)
        return ratings

    def get_similar_users(self):
        sim = self.df.loc[self.df['movie_id'] == np.int64(self.movie_id)]
        sim_list = (sim['customer_id'].head(-1).values).tolist()
        sim_ratings = (sim['rating'].head(-1).values).tolist()
        #print(sim_list)
        #print(sim_ratings)
        sim_dict = dict(zip(sim_list, sim_ratings))
        #print(sim_dict)
        return sim_dict

    def compute_similarity(self, feature_vector):
        #print(self.my_feature_vector)
        #print(feature_vector)
        x = np.array(self.my_feature_vector)
        y = np.array(feature_vector)
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        return cosine_similarity(x, y)

    def compute_weighted_rating(self, user):
        feat_vector = self.get_feature_vector(str(user))
        sim_score = self.compute_similarity(feat_vector)
        #print("similarity:", sim_score[0][0])
        #print(self.similar_users[user])
        w_rating = self.similar_users[user] * sim_score[0][0]
        print("weighted rating:", w_rating, "rating:", self.similar_users[user], "similarity score:", sim_score[0][0])
        #return sim_score[0][0]
        return w_rating

    def main(self):
        # TODO need to pass movie_id and customer_id
        self.df = self.read_train_data()
        self.num_movies = self.get_num_movies()
        self.my_feature_vector = self.get_feature_vector(self.customer_id)
        print(self.my_feature_vector)

        self.similar_users = self.get_similar_users()

        start = time.time()
        num_cores = multiprocessing.cpu_count()
        print('num_cores:', num_cores)
        res = Parallel(n_jobs=num_cores)(delayed(self.compute_weighted_rating)(i) for i in list(self.similar_users))
        end = time.time()
        print("time elapsed:", end - start)

        '''start = time.time()
        pool = Pool()
        res = pool.map(self.compute_weighted_rating, list(self.similar_users))
        pool.close()
        pool.join()
        end = time.time()
        print("time:", end - start)'''

        '''self.w_ratings = []
        start = time.time()
        for user, rating in self.similar_users.items():
            w_rating = self.compute_weighted_rating(user)
            self.w_ratings.append(w_rating)
        end = time.time()
        print("time:", end - start)
        res = self.w_ratings'''

        print("predicted rating:")
        print((sum(res) / len(res)))



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    recommender = NetflixRecommender()
    recommender.main()
