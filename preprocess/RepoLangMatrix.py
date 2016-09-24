from collections import defaultdict
from scipy.sparse import csc_matrix
import numpy as np
import json

class RepoLangMatrix(object):

    def __init__(self):
        self.lang_indexes = None
        self.repo_indexes = None
        self.repo_lang_count = None
        self.matrix = None

    def fit(self, repo_lang_dict):
        lang_indexes = {}
        repo_indexes = {}
        repo_lang_count = defaultdict(int)

        lang_cnt = 0
        repo_cnt = 0

        for repo_name, lang_list in repo_lang_dict.items():
            if repo_name not in repo_indexes:
                repo_indexes[repo_name] = repo_cnt
                repo_cnt += 1

            if not lang_list or 'message' in lang_list:
                continue

            for lang_name in lang_list:
                if lang_name not in lang_indexes:
                    lang_indexes[lang_name] = lang_cnt
                    lang_cnt += 1

                lang_idx = lang_indexes[lang_name]
                repo_idx = repo_indexes[repo_name]

                #repo_lang_count[(repo_idx, lang_idx)] += lang_list[lang_name]
                repo_lang_count[(repo_idx, lang_idx)] = 1

        self.lang_indexes = lang_indexes
        self.repo_indexes = repo_indexes
        self.repo_lang_count = repo_lang_count
        self.matrix = self.get_matrix()

    def get_matrix(self):
        repo_lang_pairs = self.repo_lang_count.keys()
        row_index, col_index = list(zip(*repo_lang_pairs))

        repo_num = len(self.repo_indexes)
        lang_num = len(self.lang_indexes)

        repo_lang_matrix = csc_matrix(
            (list(self.repo_lang_count.values()), (row_index, col_index)),
            shape=(repo_num, lang_num))

        return repo_lang_matrix

    def get_normalized_matrix(self):
        total_bytes_of_repos = self.matrix.sum(axis=1).clip(1e-15)
        return self.matrix/np.tile(total_bytes_of_repos, (1, self.matrix.shape[1]))
