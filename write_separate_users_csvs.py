import numpy as np
import pandas as pd
import os
import operator
import csv

from scipy import sparse

FILEPATH_MOVIE_TITLES = "./netflix-prize-data/movie_titles.csv"
FILEPATH_COMBINED_DATA_1 = "./netflix-prize-data/combined_data_1.txt"
FILEPATH_COMBINED_DATA_2 = "./netflix-prize-data/combined_data_2.txt"
FILEPATH_COMBINED_DATA_3 = "./netflix-prize-data/combined_data_3.txt"
FILEPATH_COMBINED_DATA_4 = "./netflix-prize-data/combined_data_4.txt"
FILEPATH_PROBE = "./netflix-prize-data/probe.txt"
FILEPATH_QUALIFYING = "./netflix-prize-data/qualifying.txt"


# Read Rating data into db ##############################################################

lines = []
datas = []
files = [FILEPATH_COMBINED_DATA_1, FILEPATH_COMBINED_DATA_2, FILEPATH_COMBINED_DATA_3, FILEPATH_COMBINED_DATA_4]
for folder in files:
	lines.clear()
	with open(folder) as file:
		lines = file.read().splitlines()
	datas = datas + lines
	print("done reading ", folder)

customers = {}
num_ratings = {}
num_movies = 0
customer_id_max = 0

movie_id = 0
for index, line in enumerate(datas):
    if (line.endswith(':')):
        movie_id = line.replace(':', '')
        num_ratings[movie_id] = 0
        num_movies += 1
    else:
        datas[index] = datas[index] + ":" + movie_id
        data = line.split(",")
        # TODO maybe append movie_id to end of each line
        if data[0] in customers:
            customers[data[0]] +=1
        else:
            customers[data[0]] = 1
        num_ratings[movie_id] +=1
        if int(data[0]) > customer_id_max:
            customer_id_max = int(data[0])

#print(datas[:10])
print("largest customer id:", customer_id_max)
print("total num customers:", len(customers))
print("total num_movies:", num_movies)

customers_list = {k: customers[k] for k in list(customers)[:10]}
print(customers_list)
sorted_customers = sorted(customers.items(), key=operator.itemgetter(1), reverse=True)
print(sorted_customers[:10])

ratings_list = {k: num_ratings[k] for k in list(num_ratings)[:10]}
print(ratings_list)
sorted_num_ratings = sorted(num_ratings.items(), key=operator.itemgetter(1), reverse=True)
print(sorted_num_ratings[:10])

probe_datas = []
similar_users = []
similar_users_ratings = {}
with open(FILEPATH_PROBE) as file:
    probe_datas = file.read().splitlines()
print("done reading probe dataset")
probe_movie_id = 0
index_start = 0
for index, line in enumerate(probe_datas):
    #print(line)
    similar_users_ratings.clear()
    if (line.endswith(':')):
        movie_id = line.replace(':', '')
        index_start = datas.index(line)
        print("index_start:", index_start)
    else:
        probe_datas[index] = probe_datas[index] + ":" + movie_id
        probe_data = line.split(",")
        user_id = probe_data[0]
        query = user_id + ','
        all_ratings = list(filter(lambda x: x.startswith(query), datas))
        # TODO get all the people who watched movie_id add them to similar users
        # then get average rating
        print(line)
        index_start+=1
        row = datas[index_start]
        while not row.endswith(':'):
            #print(row)
            row_ = row.split(',')
            similar_users_ratings[row_[0]] = int(row_[1])
            index_start+=1
            row = datas[index_start]
        # TODO
        for key, value in similar_users_ratings.items():
            query = ':' + movie_id
            #print(list(filter(lambda x: x.endswith(query), datas)))
            all_user_ratings = list(filter(lambda x: x.endswith(query), datas))
            #print(all_user_ratings)
            for user_rating in all_user_ratings:
                rating_vector = [0] * num_movies
                row_ = user_rating.split(',')
                query = row_[0] + ','
                all_row_ratings = list(filter(lambda x: x.startswith(query), datas))
                for r in all_row_ratings:
                    r_ = r.split(':')
                    r_info = r_[0].split(',')
                    rating_vector[r_[1]] = r_info[1]
                print(rating_vector)

        #print(similar_users_ratings)
        print("average rating:", (sum(similar_users_ratings.values())) / len(similar_users_ratings))
