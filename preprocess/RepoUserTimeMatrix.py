from collections import defaultdict
from scipy.sparse import csc_matrix
import numpy as np

class RepoUserTimeMatrix(object):

    def __init__(self, user_indexes, repo_indexes):
        self.user_indexes = user_indexes
        self.repo_indexes = repo_indexes
        self.repo_user_time = None
        self.matrix = None

    def fit(self, yrs, keep_event):
        user_indexes = {}
        repo_indexes = {}
        repo_user_time = defaultdict(float)

        for yr in yrs:
            print('parsing {}'.format(yr))
            path = "/mnt/disk2/georgewang/dataset/data.{}".format(yr)
            with open(path, "r") as reader:
                lines = reader.read().splitlines()

            for line in lines:
                line = line.split()

                if len(line) < 4:
                    continue

                timestamp, event_name, user_name, repo_name = line[0], line[1], line[2], line[3]

                if event_name not in keep_event:
                    continue

                if user_name not in self.user_indexes:
                    continue

                if repo_name not in self.repo_indexes:
                    continue

                user_idx = self.user_indexes[user_name]
                repo_idx = self.repo_indexes[repo_name]

                if (repo_idx, user_idx) in repo_user_time:
                    continue

                repo_user_time[(repo_idx, user_idx)] = float(timestamp)

        self.repo_user_time = repo_user_time
        self.matrix = self.get_matrix()

    def get_matrix(self):
        repo_user_pairs = self.repo_user_time.keys()
        row_index, col_index = list(zip(*repo_user_pairs))

        repo_num = len(self.repo_indexes)
        user_num = len(self.user_indexes)

        repo_user_matrix = csc_matrix(
            (list(self.repo_user_time.values()), (row_index, col_index)),
            shape=(repo_num, user_num))

        return repo_user_matrix
