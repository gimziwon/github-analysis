from collections import defaultdict
from scipy.sparse import csc_matrix
import numpy as np
class UserRepoMatrix(object):
    def __init__(self):
        self.user_indexes = None
        self.repo_indexes = None
        self.user_repo_count = None
        self.matrix = None

    def fit(self, yrs, keep_event):
        user_indexes = {}
        repo_indexes = {}
        user_repo_count = defaultdict(int)

        user_cnt = 0
        repo_cnt = 0
        for yr in yrs:
            print('parsing {}'.format(yr))
            path = "/mnt/disk2/processed_github/data.{}".format(yr)
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

                user_repo_count[(user_idx, repo_idx)] += 1

        self.user_indexes = user_indexes
        self.repo_indexes = repo_indexes
        self.user_repo_count = user_repo_count
        self.matrix = self.get_matrix()

    def get_matrix(self):
        user_repo_pairs = self.user_repo_count.keys()
        row_index, col_index = list(zip(*user_repo_pairs))

        repo_num = len(self.repo_indexes)
        user_num = len(self.user_indexes)

        user_repo_matrix = csc_matrix(
            (list(self.user_repo_count.values()), (row_index, col_index)),
            shape=(user_num, repo_num))

        return user_repo_matrix

    def filter_by_repo(self, user_threshold, repo_threshold):
        matrix = self.matrix.copy()

        user_cnt_of_repos = matrix.getnnz(axis=0)
        matrix = matrix[:, user_cnt_of_repos > repo_threshold]

        repo_cnt_of_users = matrix.getnnz(axis=1)
        matrix = matrix[repo_cnt_of_users > user_threshold, :]

        retained_row_indexes = np.where(repo_cnt_of_users > user_threshold)
        retained_col_indexes = np.where(user_cnt_of_repos > repo_threshold)

        new_row_indexes = UserRepoMatrix.get_new_indexes(
            self.user_indexes, retained_row_indexes[0])
        new_col_indexes = UserRepoMatrix.get_new_indexes(
            self.repo_indexes, retained_col_indexes[0])

        return matrix, new_row_indexes, new_col_indexes

    def filter(self, user_threshold, repo_threshold):
        matrix = self.matrix.copy()

        user_cnt_of_repos = matrix.getnnz(axis=0)
        repo_cnt_of_users = matrix.getnnz(axis=1)

        matrix = matrix[:, user_cnt_of_repos > repo_threshold]
        matrix = matrix[repo_cnt_of_users > user_threshold, :]

        retained_row_indexes = np.where(repo_cnt_of_users > user_threshold)
        retained_col_indexes = np.where(user_cnt_of_repos > repo_threshold)

        new_row_indexes = UserRepoMatrix.get_new_indexes(
            self.user_indexes, retained_row_indexes[0])
        new_col_indexes = UserRepoMatrix.get_new_indexes(
            self.repo_indexes, retained_col_indexes[0])

        return matrix, new_row_indexes, new_col_indexes

    @classmethod
    def get_normalized_matrix(cls, matrix):
        total_events_of_repos = matrix.sum(axis=0).clip(1e-15)
        return matrix/np.tile(total_events_of_repos, (matrix.shape[0], 1))

    @classmethod
    def get_new_indexes(cls, indexes, retained_indexes):
        num = 0
        new_indexes = {}
        for key, val in indexes.items():
            if val in retained_indexes:
                new_indexes[key] = num
                num += 1
        return new_indexes
