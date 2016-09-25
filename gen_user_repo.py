import sys
import pickle
import json
import numpy as np
import pandas as pd
from preprocess.RepoUserMatrix import RepoUserMatrix

def main():
    repo_user_matrix = pickle.load(open('/mnt/disk2/georgewang/obj/repo_user_matrix.pkl', 'rb'))
    user_indexes = pickle.load(open('/mnt/disk2/georgewang/result/user_indexes.pkl', 'rb'))
    repo_indexes = pickle.load(open('/mnt/disk2/georgewang/result/repo_indexes.pkl', 'rb'))

    user_list = np.array(sorted(user_indexes, key=user_indexes.get))
    repo_list = np.array(sorted(repo_indexes, key=repo_indexes.get))

    matrix = get_matrix_by_indexes(repo_user_matrix, repo_list, user_list)
    repo_userlist = {}
    for i in range(matrix.shape[0]):
        repo = matrix[i, :]
        attend_user_indexes = repo.nonzero()[1]

        repo_name = repo_list[i]
        users_name = user_list[attend_user_indexes]

        repo_userlist[repo_name] = users_name.tolist()

    with open('/mnt/disk2/georgewang/dataset/repo_userlist.json', 'w') as fp:
        json.dump(repo_userlist, fp)

def get_matrix_by_indexes(repo_user_matrix, repo_list, user_list):

    repo_indexes = [repo_user_matrix.repo_indexes[repo] for repo in repo_list]
    user_indexes = [repo_user_matrix.user_indexes[user] for user in user_list]

    matrix = repo_user_matrix.matrix.copy()
    matrix = matrix[repo_indexes, :]
    matrix = matrix[:, user_indexes]
    return matrix

if __name__ == "__main__":
    main()


