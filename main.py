import sys
import pickle
import numpy as np
from collections import Counter

from preprocess.parser import parse_matrix
from preprocess.parser import parse_lang
from preprocess.parser import transform_to_X_y
from preprocess.UserRepoMatrix import UserRepoMatrix

from sklearn.cluster import KMeans


def main():

    # load language of each repo
    repo_lang = parse_lang()
    print('repo_lang', len(repo_lang))

    # load matrix
    user_repo_matrix = pickle.load(open('/mnt/disk2/georgewang/user_repo_matrix.keep', 'rb'))
    print('user_repo_matrix', user_repo_matrix.matrix.shape)

    thresholds = {'user_threshold': 5, 'repo_threshold': 100}
    #matrix, user_indexes, repo_indexes = user_repo_matrix.filter(**thresholds)
    matrix, user_indexes, repo_indexes \
        = user_repo_matrix.filter_by_repo(**thresholds)
    matrix = user_repo_matrix.get_normalized_matrix(matrix)
    print('matrix', matrix.shape)

    for C in range(-3, 4):
        for reduced_dim in [50, 100, 300, 500]:
            params = \
                {'predict_model': {'model': 'LogisticRegression', 'params': {'C': 10**C, 'n_jobs': 5}},
                 'reduction_model': {'model': 'TruncatedSVD', 'params': {'n_components': reduced_dim}}
                }

            X, y = transform_to_X_y(matrix, repo_lang, repo_indexes, params)
            print("X", X.shape)
            print("y", y.shape)
            print("unique_y", np.unique(y))

            for cluster_num in [5, 10, 15, 20]:
                #do clustering
                kmeans = KMeans(init='k-means++', n_clusters=8, n_jobs=8, max_iter=1000)
                pred_labels = kmeans.fit_predict(X)
                print('pred_labels', len(pred_labels))
                print(Counter(pred_labels))

                with open("/mnt/disk2/georgewang/C_{}_dim_{}_cluster_{}.txt"
                          .format(C, reduced_dim, cluster_num), "w") as writer:
                    writer.write(str(Counter(pred_labels)) + '\n')
                    for repo_name, repo_index in repo_indexes.items():
                        writer.write("{}\t{}\n".format(repo_name, pred_labels[repo_index]))

if __name__ == "__main__":
    main()

