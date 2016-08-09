from collections import defaultdict
from scipy.sparse import csc_matrix
import numpy as np
class RepoUserMatrix(object):
    def __init__(self):
        self.user_indexes = None
        self.repo_indexes = None
        self.repo_user_count = None
        self.matrix = None

    def fit(self, yrs, keep_event):
        user_indexes = {}
        repo_indexes = {}
        repo_user_count = defaultdict(int)

        user_cnt = 0
        repo_cnt = 0
        for yr in yrs:
            print('parsing {}'.format(yr))
            path = "/mnt/disk2/georgewang/dataset/data.{}".format(yr)
            with open(path, "r") as reader:
                lines = reader.read().splitlines()

            for line in lines:
                line = line.split()

                if len(line) < 4:
                    continue

                event_name, user_name, repo_name = line[1], line[2], line[3]

                if event_name not in keep_event:
                    continue

                if user_name not in user_indexes:
                    user_indexes[user_name] = user_cnt
                    user_cnt += 1

                if repo_name not in repo_indexes:
                    repo_indexes[repo_name] = repo_cnt
                    repo_cnt += 1

                user_idx = user_indexes[user_name]
                repo_idx = repo_indexes[repo_name]

                repo_user_count[(repo_idx, user_idx)] += 1

        self.user_indexes = user_indexes
        self.repo_indexes = repo_indexes
        self.repo_user_count = repo_user_count
        self.matrix = self.get_matrix()

    def get_matrix(self):
        repo_user_pairs = self.repo_user_count.keys()
        row_index, col_index = list(zip(*repo_user_pairs))

        repo_num = len(self.repo_indexes)
        user_num = len(self.user_indexes)

        repo_user_matrix = csc_matrix(
            (list(self.repo_user_count.values()), (row_index, col_index)),
            shape=(repo_num, user_num))

        return repo_user_matrix

    def filter(self, repo_threshold=100, user_threshold=5, event_threshold=10):
        matrix = self.matrix.copy()

        user_cnt_of_repos = matrix.getnnz(axis=1)
        retained_row_indexes = np.where(user_cnt_of_repos > repo_threshold)[0]
        matrix = matrix[retained_row_indexes, :]

        retained_col_indexes = np.where(np.any(matrix.toarray() > event_threshold, axis=0))[0]
        matrix = matrix[:, retained_col_indexes]

        return matrix, retained_row_indexes, retained_col_indexes

    @classmethod
    def get_normalized_matrix(cls, matrix):
        total_events_of_repos = matrix.sum(axis=1).clip(1e-15)
        return matrix/np.tile(total_events_of_repos, (1, matrix.shape[1]))

    @classmethod
    def get_new_indexes(cls, indexes, retained_indexes):
        num = 0
        new_indexes = {}
        for key, val in sorted(indexes.items(), key=lambda x: x[1]):
            if val in retained_indexes:
                new_indexes[key] = num
                num += 1
        return new_indexes
