'''グリッドサーチをSVMでやる'''
# coding: utf-8
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import io
import sys


def svm_grid(X, y, scorer='accuracy', k=5):

    pipe_svm = Pipeline([
        ('scl', StandardScaler()),
        ('svm', SVC(random_state=1))
    ])

    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    param_grid = [
        {
            'svm__C': param_range,
            'svm__kernel': ['linear']
        },
        {
            'svm__C': param_range,
            'svm__gamma': param_range,
            'svm__kernel': ['rbf']
        }
    ]

    grid = GridSearchCV(
        estimator=pipe_svm,
        param_grid=param_grid,
        scoring=scorer,
        cv=k,
        n_jobs=-1
    )

    # 表示しなくていいエラーが出るので，'stderr'をテキストバッファに出力
    # ただし，他のエラーの場合に気づけないので注意
    sys.stderr = io.StringIO()

    gs = grid.fit(X, y)

    # エラーの出力を元に戻す
    sys.stderr = sys.__stderr__

    return gs
