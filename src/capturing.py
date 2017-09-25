# coding: utf-8

"""
Convert the video to an image of each interval and save it
"""

import cv2
import glob
import os
import shutil
from tqdm import tqdm

# Module to convert video to array
from equipment.capture_video import capture_video

# 読み込み・書き出し用のライブラリ
import common.save_and_load as SL

# Read content of status.yml
status = SL.load_yaml('status.yml')
sec = status['Interval']    # 切り出し間隔
resizing = status['resizing']   # 画像のリサイズをするかしないか
img_width = status['img_width'] # 画像の横幅
img_height = status['img_height'] # 画像の縦幅
file_path = status['video_root']    # 映像ファイルディレクトリまでのパス
save_path = status['img_path'] # 画像を保存するディレクトリ


def main():

    """   mp4ファイルまでのパスと，その動画名を取得   """
    # 'file_path'直下のすべてのmp4ファイルまでのパスを取得する
    files = glob.glob(file_path + '*.mp4')
    names = list(map(SL.get_filename, files))


    """   保存用のディレクトリを作成する（すでにある場合は中のデータごと削除）   """
    make_directory(save_path)


    """   映像を配列に変換し，jpgで保存する   """
    # 進捗を表示する場合は'tqdm'で囲う（なくてもいい）
    files = tqdm(files)
    for file, name in zip(files, names):

        # 映像を配列に変換
        video = capture_video(
            file, sec=sec, resizing=resizing, width=img_width, height=img_height
        )

        # 保存先のディレクトリを生成する
        save_img_path = save_path + name + '/'
        make_directory(save_img_path)

        # 画像を保存する（名前を'00*.jpg'のようにパディングする）
        for i, img in enumerate(video):
            save_name = '%03d.jpg' % i
            save_name = save_img_path + save_name
            cv2.imwrite(save_name, img)

def make_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

if __name__ == '__main__':
    main()
