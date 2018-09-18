import numpy as np
import pandas as pd
import os
import operator
import csv

FILEPATH_MOVIE_TITLES = "./netflix-prize-data/movie_titles.csv"
FILEPATH_COMBINED_DATA_1 = "./netflix-prize-data/combined_data_1.txt"
FILEPATH_COMBINED_DATA_2 = "./netflix-prize-data/combined_data_2.txt"
FILEPATH_COMBINED_DATA_3 = "./netflix-prize-data/combined_data_3.txt"
FILEPATH_COMBINED_DATA_4 = "./netflix-prize-data/combined_data_4.txt"
FILEPATH_PROBE = "./netflix-prize-data/probe.txt"
FILEPATH_QUALIFYING = "./netflix-prize-data/qualifying.txt"

CLEAN_1 = "./netflix_data_clean/clean_data_1.csv"
CLEAN_2 = "./netflix_data_clean/clean_data_2.csv"
CLEAN_3 = "./netflix_data_clean/clean_data_3.csv"
CLEAN_4 = "./netflix_data_clean/clean_data_4.csv"

files = [FILEPATH_COMBINED_DATA_1, FILEPATH_COMBINED_DATA_2, FILEPATH_COMBINED_DATA_3, FILEPATH_COMBINED_DATA_4]
writes = [CLEAN_1, CLEAN_2, CLEAN_3, CLEAN_4]
for folder, write in zip(files, writes):
    lines = []
    movie_id = 0
    with open(folder) as file:
        lines = file.read().splitlines()
        for index, line in enumerate(lines):
            if line.endswith(':'):
                movie_id = line.replace(':', '')
            else:
                lines[index] = movie_id + ',' + lines[index]
    matching = [s for s in lines if not ':' in s]
    #print(matching[:100])
    with open(write, 'w') as outfile:
        wr = csv.writer(outfile, delimiter=',')
        wr.writerow(['movie_id', 'customer_id', 'rating', 'date'])
        wr.writerows([x.split(',') for x in matching])
    print("finished writing:", write)
