import numpy as np
import os
import pickle
import sys
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC


if __name__ == '__main__':
	data = np.load(sys.argv[1])
	X, y = data['arr_0'], data['arr_1']
	le = LabelEncoder().fit(y)
	model = SVC(kernel='linear', probability=True, C=1)
	model.fit(Normalizer('l2').transform(X), y)

	filename = '{}/classifier.pkl'.format(sys.argv[2])
	print('Saving classifier to: {}'.format(filename))
	with open(filename, 'wb') as f:
		pickle.dump((le, model), f)
