import sys
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression

def parse_matrix():
    params = {'yrs': [2011, 2012, 2013, 2014, 2015, 2016],
              'keep_event': set(['PushEvent', 'PullRequestEvent', 'CommitCommentEvent'])}

    user_repo_matrix = RepoUserMatrix()
    user_repo_matrix.fit(**params)

    return user_repo_matrix

def parse_lang():
    with open("/mnt/disk2/georgewang/dataset/repo.lang", "r") as reader:
        lines = reader.read().splitlines()

    repo_lang = {}
    for line in lines:
        repo_name, lang = line.split(sep='\t')
        repo_lang[repo_name] = lang

    return repo_lang

def transform_to_X_y(matrix, repo_lang, repo_list, params):
    y = []
    for repo_name in repo_list:
        if repo_name not in repo_lang:
            y.append('None')
        else:
            y.append(repo_lang[repo_name])
    y = np.array(y)

    if params['reduction_model'] == None:
        X = matrix.T
    else:
        reduction_model_class = getattr(sys.modules[__name__], params['reduction_model']['model'])
        reduction_model_params = params['reduction_model']['params']

        reduction_model = reduction_model_class(**reduction_model_params)
        X = reduction_model.fit_transform(matrix.T)

    if params['predict_model'] == None:
        X = X
    else:
        predict_model_class = getattr(sys.modules[__name__], params['predict_model']['model'])
        predict_model_params = params['predict_model']['params']
        predict_model = predict_model_class(**predict_model_params)

        predict_model.fit(X, y)
        X = predict_model.predict_proba(X)

    return X, y
