# coding : utf-8

"""
学習に使用する映像の特徴抽出
結果をcsvとして出力
"""

# 必要なライブラリ
import cv2
import pandas as pd
import glob
import os
from tqdm import tqdm
import numpy as np

# 特徴抽出ライブラリ（自作）
#from equipment.visual_analyzer import VisualAnalyzer
from equipment.analyzer_0 import VisualAnalyzer
from equipment.capture_video import capture_video

# 読み込み・書き出しのライブラリ
import common.save_and_load as SL

# 'status.yml'の内容を読み込む
status = SL.load_yaml('status.yml')
img_path = status['img_path']  # 画像を保存したディレクトリ
output_file = status['features_file']   # 出力用のファイルパス

# analyzer_ryuhei_0のパラメータ（辞書型）
colorNames = ["赤", "橙", "ピンク", "黄色", "緑", "青", "紫", "茶色", "白", "灰色", "黒"]
colors = [
    [0, 0, 255], [0, 128, 255], [255, 100, 255], [
        0, 255, 255], [0, 255, 0], [255, 0, 0],
    [255, 0, 255], [40, 70, 125], [255, 255, 255], [125, 125, 125], [0, 0, 0]
]


def main():
    """   映像（画像）配列の読み込み   """
    # ディレクトリの名前を取得する（動画IDとなる）
    names = os.listdir(img_path)
    names = list(filter(lambda x: os.path.isdir(img_path + x), names))

    # 各画像ディレクトリまでのパスを文字列として生成
    img_paths = list(map(lambda x: img_path + x + '/', names))

    print('\n   映像配列を取得中...\n')
    # 進捗を表示する場合は'tqdm'で囲う
    img_paths = tqdm(img_paths)
    videos = list(map(load_images, img_paths))

    """   VisualAnalyzer()によって特徴量を抽出する   """
    analyzer = VisualAnalyzer(colors, colorNames)  # 分析器（クラス）の定義

    # 'analyzer.extract'の結果を'feature_values'に入れる
    print('\n   特徴量を抽出中...\n')
    # 進捗を表示する場合は'tqdm'で囲う
    videos = tqdm(videos)
    feature_values = list(map(analyzer.extract, videos))

    """   特徴量をcsvに保存する   """
    # 'feature_values'は二次配列の表となるため，DataFrameに変換
    df = pd.DataFrame(feature_values, index=names,
                      columns=analyzer.feature_names)
    df.index.name = '動画ID'

    # 保存する
    SL.df_to_csv(df, output_file)


def load_images(dir_name):

    files = glob.glob(dir_name + '*.jpg')

    video = list(map(lambda x: cv2.imread(x), files))
    video = np.array(video)

    return video


if __name__ == '__main__':
    main()
