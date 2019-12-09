import numpy as np
import os
import pickle
import sys
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder, Normalizer

from detect_faces import detect_faces


if __name__ == '__main__':
	with open(sys.argv[1], 'rb') as f:
		(le, model) = pickle.load(f, encoding='latin1')

	embedder = FaceNet()
	directory = sys.argv[2]
	files = sorted(os.listdir(directory))

	for i, filename in enumerate(files):
		path = directory + filename
		faces = detect_faces(path)
		result = []

		for face in faces:
			rep = embedder.embeddings([face])
			pred = model.predict_proba(rep).ravel()
			maxI = np.argmax(pred)
			confidence = pred[maxI]
			person = le.inverse_transform([maxI])[0]
			result.append('{} ({:.2f})'.format(person, confidence))

		print(i, ', '.join(result))