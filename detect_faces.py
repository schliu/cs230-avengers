import numpy as np
import os
import sys
from mtcnn.mtcnn import MTCNN
from PIL import Image


IMAGE_SIZE = 182
detector = MTCNN()
CONF_THRESHOLD = 0.5


def detect_faces(src, min_conf=CONF_THRESHOLD):
	img = Image.open(src).convert('RGB')
	pixels = np.asarray(img)
	boxes = detector.detect_faces(pixels)
	faces = []
	confidences = []
	locations = []
	for box in boxes:
		if box['confidence'] > min_conf:
			x1, y1, width, height = box['box']
			x1, y1 = abs(x1), abs(y1)
			x2, y2 = x1 + width, y1 + height
			face = pixels[y1:y2, x1:x2]
			aligned = Image.fromarray(face).resize((IMAGE_SIZE, IMAGE_SIZE))
			faces.append(np.asarray(aligned))
			confidences.append(box['confidence'])
			locations.append([(x1, y1), (x2, y2)])
	return faces, confidences, locations, img


def load_faces(directory):
	result = []
	for filename in os.listdir(directory):
		try:
			path = directory + filename
			faces = detect_faces(path)
			result += faces
		except Exception as e:
			print('Error: {}'.format(str(e)))
	return result


def load_dataset(root):
	X =[]
	y = []
	for d in os.listdir(root):
		try:
			path = '{}/{}/'.format(root, d)
			if os.path.isdir(path):
				print('Loading: {}'.format(d))
				faces = load_faces(path)
				labels = [d for _ in range(len(faces))]
				X.extend(faces)
				y.extend(labels)
				print('Finished loading: {}'.format(d))
		except Exception as e:
			print('Error: {}'.format(str(e)))
	return np.asarray(X), np.asarray(y)


if __name__ == '__main__':
	X, y = load_dataset(sys.argv[1])
	np.savez_compressed(sys.argv[2], X, y)