import argparse
import numpy as np
import sys
import os
import pickle
from keras_facenet import FaceNet
from PIL import Image

from utils import *


IMAGE_SIZE = 182

img_path = "../downloads/ageofultron/ageofultron_32.jpg"
dir_path = "../downloads/avengers2012/avengers2012_"
model_cfg = "./cfg/yolov3-face.cfg"
model_weights = "./model-weights/yolov3-wider_16000.weights"
output_dir = "outputs/"

# Give the configuration and weight files for the model and load the network
# using them.
net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def generate_input_paths(dir, frames):
    img_paths = []
    for i in range(frames):
        img_paths.append(dir + str(i) + ".jpg")
    return img_paths


def process_image(img):
    # Verify input and prepare output file
    output_file = ''
    if not os.path.isfile(img):
        print("[!] ==> Input image file {} doesn't exist".format(img))
        sys.exit(1)
    cap = cv2.VideoCapture(img)
    output_file = img[:-4].rsplit('/')[-1] + '_yoloface.jpg'

    has_frame, frame = cap.read()

    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                 [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(get_outputs_names(net))

    # Remove the bounding boxes with low confidence
    faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)

    # # initialize the set of information we'll displaying on the frame
    # info = [
    #     ('number of faces detected', '{}'.format(len(faces)))
    # ]

    # for (i, (txt, val)) in enumerate(info):
    #     text = '{}: {}'.format(txt, val)
    #     cv2.putText(frame, text, (10, (i * 20) + 20),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)

    # cv2.imwrite(os.path.join(output_dir, output_file), frame.astype(np.uint8))

    # if (len(faces) > 0):
    #     print('{} ==> # detected faces: {}'.format(img, len(faces)))

    cap.release()

    return faces

def _main():
    # img_paths = generate_input_paths(dir_path, 138)
    # for ip in img_paths:
    #     faces = process_image(ip)
    with open(sys.argv[1], 'rb') as f:
        (le, model) = pickle.load(f, encoding='latin1')

    embedder = FaceNet()
    directory = sys.argv[2]
    files = sorted(os.listdir(directory))

    for i, filename in enumerate(files):
        path = directory + filename
        boxes = process_image(path)
        img = np.asarray(Image.open(path).convert('RGB'))

        result = []
        for box in boxes:
            x1, y1, width, height = box
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            face = img[y1:y2, x1:x2]
            face = Image.fromarray(face).resize((IMAGE_SIZE, IMAGE_SIZE))
            face = np.asarray(face)

            rep = embedder.embeddings([face])
            pred = model.predict_proba(rep).ravel()
            maxI = np.argmax(pred)
            confidence = pred[maxI]
            person = le.inverse_transform([maxI])[0]
            result.append('{} ({:.2f})'.format(person, confidence))

        print(i, ', '.join(result))


if __name__ == '__main__':
    _main()
