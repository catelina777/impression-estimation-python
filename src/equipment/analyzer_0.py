#coding: utf-8

'''
映像特徴量の抽出プログラム
上西が作った分析器
色を決めうちで入力し、そのカラーヒストグラムを特徴量にする
色には名前をつけて次元名に入れる
'''

# ライブラリの読み込み
import cv2
import numpy as np
from tqdm import tqdm

'''		VisualAnalyzer		'''


class VisualAnalyzer:

    def __init__(self, colors, color_names):
        self.colors = np.array(colors)
        self.B_sq = np.power(self.colors, 2).sum(axis=1)
        self.feature_names = color_names

    # 映像の各画像に対して特徴量を抽出する
    # 映像全体の特徴ベクトルを作る
    def extract(self, video):

        # colors = np.array(self.colors)
        # B_sq = np.power(colors, 2).sum(axis=1)

        vector_sum = np.zeros(len(self.colors))

        for img in video:
            img_h, img_w = img.shape[0], img.shape[1]
            img = img.reshape(img_h * img_w, 3)

            A_sq = np.power(img, 2).sum(axis=1)
            A_sq = A_sq.reshape(len(A_sq), 1)

            inner = np.dot(img, self.colors.T)

            dists_all = A_sq - (2 * inner) + self.B_sq

            vector = np.zeros(len(self.colors))
            idxs = np.argmin(dists_all, axis=1)
            for i in range(len(vector)):
                counts = len(np.where(idxs == i)[0])
                vector[i] = counts

            vector_sum = vector_sum + vector

        vector_mean = vector_sum / len(video)
        vector_norm = vector_mean / (video[0].shape[0] * video[0].shape[1])
        return vector_norm

    def get_feature_names(self):
        names = self.thresholdColorNames
        return names


def main():
    from capture_video import CaptureVideo
    # docker_file_path = '/usr/workspace/visual_source/sm12253877.mp4'
    # video = CaptureVideo(docker_file_path)
    file_path = 'sample/sample.mp4'
    video = CaptureVideo(file_path, 1)

    color_names = ["赤", "橙", "ピンク", "黄色", "緑", "青", "紫", "茶色", "白", "灰色", "黒"]

    colors = [
        [0, 0, 255], [0, 128, 255], [255, 100, 255], [
            0, 255, 255], [0, 255, 0], [255, 0, 0],
        [255, 0, 255], [40, 70, 125], [
                255, 255, 255], [125, 125, 125], [0, 0, 0]
    ]

    analyzer = VisualAnalyzer(colors, color_names)

    feature_values = analyzer.extract(video)
    print(feature_values)
    print(analyzer.feature_names)
    print(sum(feature_values))


if __name__ == '__main__':
    main()
