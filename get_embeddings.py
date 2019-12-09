import numpy as np
import sys
from keras.models import load_model
from keras_facenet import FaceNet


BATCH_SIZE = 1000
embedder = FaceNet()


if __name__ == '__main__':
	data = np.load(sys.argv[1])
	X, y = data['arr_0'], data['arr_1']
	print('Loaded X: {}, y: {}'.format(X.shape, y.shape))
	embeddings = []
	for i in range(0, X.shape[0], BATCH_SIZE):
		batch = X[i : i+BATCH_SIZE]
		print(batch.shape)
		batch_emb = np.asarray(embedder.embeddings(batch))
		embeddings.extend(batch_emb)
	np.savez_compressed(sys.argv[2], embeddings, y)