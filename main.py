import sys, os
import pickle
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

from preprocess.UserRepoMatrix import UserRepoMatrix

def main():

    user_repo_matrix = pickle.load(open('/mnt/disk2/georgewang/user_repo_matrix.obj', 'rb'))

    thresholds = {'user_threshold': 100, 'repo_threshold': 100}
    matrix, user_indexes, repo_indexes = user_repo_matrix.filter(**thresholds)

    import ipdb; ipdb.set_trace()
    # do SVD
    svd = TruncatedSVD(n_components=100)
    reduced_matrix = svd.fit_transform(matrix.T)

    #do clustering
    kmeans = KMeans(init='k-means++', n_clusters=10, n_jobs=8)
    pred_labels = kmeans.fit_predict(reduced_matrix)

    print(Counter(pred_labels))

    with open("/mnt/disk2/georgewang/repo_{}_user_{}".format(
        thresholds['repo_threshold'], thresholds['user_threshold']),
              "w") as writer:
        for repo_name, repo_index in repo_indexes.items():
            writer.write("{}\t{}\n".format(repo_name, pred_labels[repo_index]))

def preprocess():
    params = {'yrs': [2011, 2012, 2013, 2014],
              'ignore_event': set(['ForkEvent', 'WatchEvent', 'FollowEvent'])}
    user_repo_matrix = UserRepoMatrix()
    user_repo_matrix.fit(**params)

    thresholds = {'user_threshold': 100, 'repo_threshold': 100}
    matrix, user_indexes, repo_indexes = user_repo_matrix.filter(**thresholds)
    return matrix, user_indexes, repo_indexes

if __name__ == "__main__":
    main()
