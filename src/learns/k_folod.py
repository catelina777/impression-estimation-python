'''交差検定により性能を評価する'''

# coding: utf-8

import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix


def cross_validation(X, y, k, clf, class_num=2):

    kfold = StratifiedKFold(n_splits=k, random_state=1)

    # 正解率の配列
    accuracy = np.array([])

    # 適合率・再現率・F値・テストサイズの配列
    precision = np.empty((0, class_num))
    recall = np.empty((0, class_num))
    f1 = np.empty((0, class_num))
    support = np.empty((0, class_num))

    # 混同行列
    confmat = 0

    for train, test in kfold.split(X, y):

        clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        y_true = y[test]

        # 正解率を追加する
        accuracy = np.append(accuracy, accuracy_score(y_true, y_pred))

        # 適合率・再現率・F値・テストサイズを追加する
        prfs_score = precision_recall_fscore_support(
            y_true, y_pred
        )
        precision = np.vstack((precision, prfs_score[0]))
        recall = np.vstack((recall, prfs_score[1]))
        f1 = np.vstack((f1, prfs_score[2]))
        support = np.vstack((support, prfs_score[3]))

        # 混同行列を追加する
        mat = confusion_matrix(y_true, y_pred)
        confmat += mat

    scores = {
        'accuracy': accuracy.mean(),
        'precision': list(precision.mean(axis=0)),
        'recall': list(recall.mean(axis=0)),
        'f1': list(f1.mean(axis=0)),
        'support': list(np.round(support.mean(axis=0)))
    }

    confmat = np.abs(confmat // k)

    return scores, confmat
