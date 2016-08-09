from collections import defaultdict
from scipy.sparse import csc_matrix
import numpy as np
class RepoLangMatrix(object):
    def __init__(self):
        self.lang_indexes = None
        self.repo_indexes = None
        self.repo_lang_count = None
        self.repo_lang_dict = None
        self.matrix = None

    def fit(self):
        lang_indexes = {}
        repo_indexes = {}
        repo_lang_dict = {}
        repo_lang_count = defaultdict(int)

        lang_cnt = 0
        repo_cnt = 0

        path = "/mnt/disk2/georgewang/dataset/repo.lang"
        with open(path, "r") as reader:
            lines = reader.read().splitlines()

        print("line number:", len(lines))

        for line in lines:
            line = line.split()

            if len(line) < 2:
                continue

            repo_name, lang_name = line[0], line[1]

            repo_lang_dict[repo_name] = lang_name

            if lang_name not in lang_indexes:
                lang_indexes[lang_name] = lang_cnt
                lang_cnt += 1

            if repo_name not in repo_indexes:
                repo_indexes[repo_name] = repo_cnt
                repo_cnt += 1

            lang_idx = lang_indexes[lang_name]
            repo_idx = repo_indexes[repo_name]

            repo_lang_count[(repo_idx, lang_idx)] = 1

        self.lang_indexes = lang_indexes
        self.repo_indexes = repo_indexes
        self.repo_lang_dict = repo_lang_dict
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
