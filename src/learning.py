# coding: utf-8
'''映像の印象推定の機械学習'''

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.metrics import make_scorer, accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score
from learns.svm_grid import svm_grid
from learns.k_folod import cross_validation

values_file = '../results/feature_values.csv'
mood_file = '../data/dataset_visual.tsv'
output_file = '../results/result.csv'

moods = [
    'C1(堂々)', 'C2(元気が出る)', 'C3(切ない)', 'C4(激しい)',
    'C5(滑稽な)', 'C6(かわいい)', 'Valence', 'Arousal'
]
n_split = 5


def main():
    # 特徴量データの取り出し
    with open(values_file, 'r') as f:
        df_values = pd.read_csv(f, index_col=0)

    # 印象データの取り出し
    with open(mood_file, 'r', encoding='utf_8') as f:
        df_moods = pd.read_csv(f, index_col=0, sep='\t')

    # 印象データを整形（Valence・Arousalの値を+2）
    df_moods = df_moods.loc[df_values.index, moods]
    df_moods[['Valence', 'Arousal']] += 3

    output = {}

    for mood in tqdm(moods):

        # 印象値にラベルをつける
        df_mood = df_moods[mood]
        df_label = labeling(df_mood, 4.0, 2.0)
        df = pd.concat([df_label, df_values], axis=1, join='inner')

        # 特徴量とラベルを取り出す
        X = df.iloc[:, 1:].values.astype(float)
        y = df.iloc[:, 0].values

        # パラメータのチューニング
        gs = svm_grid(X, y)
        # scorer = make_scorer(precision_score, pos_label=1)
        # gs = svm_grid(X, y, scorer=scorer)
        clf = gs.best_estimator_

        # 交差検定
        scores, confmat = cross_validation(X, y, n_split, clf)

        pl = 1
        # 結果の整形
        result = {
            '正解率': scores['accuracy'],
            '適合率': scores['precision'][pl],
            '再現率': scores['recall'][pl],
            'F値': scores['f1'][pl],
            'データ数': len(y),
            'テストサイズ': scores['support'][pl],
            '分割数': n_split
        }
        output[mood] = result

    df_out = pd.DataFrame.from_dict(output, orient='index')
    df_out = df_out.ix[moods, :]

    with open(output_file, 'w', encoding='utf_8') as f:
        df_out.to_csv(f)


# データをラベルに変換する（positive or negative）
# ptはposの閾値，ntはnegの閾値
def labeling(df, pt, nt):

    df_pos = df[df >= pt].sort_values(ascending=False)
    df_neg = df[df <= nt].sort_values()

    length = min(len(df_pos), len(df_neg))
    df_pos = df_pos[:length]
    df_neg = df_neg[:length]

    df_pos[:] = 1
    df_neg[:] = 0

    df_label = pd.concat([df_pos, df_neg])

    return df_label


if __name__ == '__main__':
    main()
