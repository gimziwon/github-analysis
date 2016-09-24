import sys
import pickle
import json
import numpy as np
from collections import Counter

from preprocess.parser import parse_matrix
from preprocess.parser import parse_lang
from preprocess.parser import transform_to_X_y
from preprocess.RepoUserMatrix import RepoUserMatrix
from preprocess.RepoLangMatrix import RepoLangMatrix
from preprocess.RepoUserTimeMatrix import RepoUserTimeMatrix

from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

def main_repo():

    repo_user_matrix = pickle.load(open('/mnt/disk2/georgewang/obj/repo_user_matrix.pkl', 'rb'))
    print("repo_user_matrix", repo_user_matrix.matrix.shape)

    threshold = {}
    threshold['user_threshold'] = 5
    threshold['repo_threshold'] = 100
    threshold['event_threshold'] = 10

    matrix, retained_row_indexes, retained_col_indexes \
        = repo_user_matrix.filter(**threshold)
    print("after filtering...", matrix.shape)
    repo_indexes = RepoUserMatrix.get_new_indexes(repo_user_matrix.repo_indexes, retained_row_indexes)
    user_indexes = RepoUserMatrix.get_new_indexes(repo_user_matrix.user_indexes, retained_col_indexes)

    normalized_matrix = RepoUserMatrix.get_normalized_matrix(matrix)

    X = normalized_matrix
    latent_vector = get_latent_vector(X)
    repo_sim_matrix = latent_vector.dot(latent_vector.T)

    pickle.dump(repo_sim_matrix, open('/mnt/disk2/georgewang/result/repo_sim_matrix.pkl', 'wb'))
    pickle.dump(repo_indexes, open('/mnt/disk2/georgewang/result/repo_indexes.pkl', 'wb'))
    pickle.dump(user_indexes, open('/mnt/disk2/georgewang/result/user_indexes.pkl', 'wb'))
    pickle.dump(latent_vector, open('/mnt/disk2/georgewang/result/latent_vector.pkl', 'wb'))
    import ipdb; ipdb.set_trace()

def main_lang():
    repo_lang_matrix = RepoLangMatrix()

    path = '/mnt/disk2/ejhsu/github/repoLangs.json'
    with open(path, 'r') as reader:
        json_obj = json.load(reader)

    repo_lang_matrix.fit(json_obj)

    X = repo_lang_matrix.matrix
    latent_vector = get_latent_vector(X.T)
    lang_sim_matrix = latent_vector.dot(latent_vector.T)

    #pickle.dump(lang_sim_matrix, open('/mnt/disk2/georgewang/result/lang_sim_matrix.pkl', 'wb'))
    #pickle.dump(repo_lang_matrix.lang_indexes, open('/mnt/disk2/georgewang/result/lang_indexes.pkl', 'wb'))
    import ipdb; ipdb.set_trace()

def main_time():

    repo_indexes = pickle.load(open('/mnt/disk2/georgewang/result/repo_indexes.pkl', 'rb'))
    repo_user_time_matrix = pickle.load(open('/mnt/disk2/georgewang/obj/repo_user_time_matrix.pkl', 'rb'))
    repo_list = sorted(repo_indexes, key=repo_indexes.get)

    fp = open('/mnt/disk2/georgewang/pseudo_label/all.txt', 'w')
    for i in range(repo_user_time_matrix.matrix.shape[0]-1):
        print(str(i))
        for j in range(i+1, repo_user_time_matrix.matrix.shape[0]):
            repo1 = repo_user_time_matrix.matrix[i, :].toarray()
            repo2 = repo_user_time_matrix.matrix[j, :].toarray()

            common_user_indexes = np.logical_and(repo1, repo2)
            common_user_n = np.sum(common_user_indexes)
            if common_user_n > 0:
                time_diff = repo2[common_user_indexes]-repo1[common_user_indexes]
                fp.write('{} {} {} {} {}\n'.format(repo_list[i], repo_list[j],
                         common_user_n, np.sum(time_diff>0), np.sum(time_diff<0)))

    import ipdb; ipdb.set_trace()

def get_top_langs(lang, matrix, lang_indexes):
    topK = 5
    lang_idx = lang_indexes[lang]
    top_sim_langs = np.argsort(matrix[lang_idx, :])[::-1][:topK]
    sorted_lang = np.array(sorted(lang_indexes, key=lang_indexes.get))
    return sorted_lang[top_sim_langs], np.sort(matrix[lang_idx, :])[::-1][:topK]

def get_latent_vector(X):
    # for language: n_components=150
    # for repo: n_components=?
    model = NMF(n_components=150, init='nndsvd', max_iter=1000, random_state=1126)
    print('NMF', model)
    model.fit(X)
    W = model.transform(X)
    H = model.components_

    normalized_matrix = normalize(W, axis=1, norm='l2')
    return normalized_matrix

def save_repo_user_matrix():
    repo_user_matrix = RepoUserMatrix()
    repo_user_matrix.fit(yrs=[2011, 2012, 2013, 2014, 2015, 2016],
                         keep_event=set(['PushEvent', 'PullRequestEvent',
                                         'CommitCommentEvent']))
    pickle.dump(repo_user_matrix, open('/mnt/disk2/georgewang/obj/repo_user_matrix2.pkl', 'wb'))

def save_repo_lang_matrix():
    repo_lang_matrix = RepoLangMatrix()
    repo_lang_matrix.fit()
    pickle.dump(repo_lang_matrix, open('/mnt/disk2/georgewang/obj/repo_lang_matrix.pkl', 'wb'))

def save_repo_user_time_matrix():
    user_indexes = pickle.load(open('/mnt/disk2/georgewang/result/user_indexes.pkl', 'rb'))
    repo_indexes = pickle.load(open('/mnt/disk2/georgewang/result/repo_indexes.pkl', 'rb'))

    repo_user_time_matrix = RepoUserTimeMatrix(user_indexes, repo_indexes)
    repo_user_time_matrix.fit(yrs=[2011, 2012, 2013, 2014, 2015, 2016],
                         keep_event=set(['PushEvent', 'PullRequestEvent',
                                         'CommitCommentEvent']))

    pickle.dump(repo_user_time_matrix, open('/mnt/disk2/georgewang/obj/repo_user_time_matrix.pkl', 'wb'))
    import ipdb; ipdb.set_trace()

def get_top_users(repo_vector, user_indexes, topK=5):

    np.set_printoptions(threshold=10)

    user_indexes = np.array(user_indexes)

    sorted_event_count = np.argsort(repo_vector).flatten()[::-1][:topK]
    return user_indexes[sorted_event_count]

if __name__ == "__main__":
    #main_repo()
    #save_repo_user_matrix()
    #main_lang()
    #save_repo_user_time_matrix()
    main_time()

