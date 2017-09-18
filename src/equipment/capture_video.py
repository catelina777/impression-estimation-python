#coding: utf-8

'''
Convert Video File to an array of images
'''

import cv2
import numpy as np

'''		CaptureVideo() 	'''
# 'cv2.VideoCapture()'を扱いやすい形にする
# 'cv2.VideoCapture'クラスを'numpy.ndarray'クラスに変換する
# 'array[時間][Y座標][X座標][B,G,R(0,1,2)]'の4次元配列
# 'resizing=True'のとき'width','height'のサイズにリサイズ（正規化）する
# 'sec'の値（秒）ごとに画像を取得
def CaptureVideo(file, sec=5, resizing=True, width=300, height=200):

	cap = cv2.VideoCapture(file)

	# Get original size when not resizing
	if resizing == False:
		width = int(cap.get(3))
		height = int(cap.get(4))

	# Prepare arrays from the same size as the image
	array = np.empty((0, height, width, 3))
	time = 0

	while(cap.isOpened()):

		# frame is an image，ret is readable / writable（false on failure）
		ret, frame = cap.read()
		if ret == False : break

		# Resize
		if resizing == True:
			frame = cv2.resize(frame, (width, height))

		# Add an image to the array
		array = np.append(array, [frame], axis=0)

		# Advance time
		time += sec * 1000
		cap.set(0, time)

	return array


def main():
	file = 'sample/sample.mp4'
	video = CaptureVideo(file)
	cv2.imwrite('sample/result.jpg', video[2])
	print(video.shape)

if __name__ == '__main__':
	main()
