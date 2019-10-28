import cv2
import argparse

def video_to_images(video, output_path):
	step = 1000 # millseconds

	vidcap = cv2.VideoCapture(video)
	success = True
	frame = 0
	success, image = vidcap.read()
	while success:
		vidcap.set(cv2.CAP_PROP_POS_MSEC, frame * 1000)
		cv2.imwrite(str(output_path) + ('_%d.jpg' % frame), image)
		frame += 1
		success, image = vidcap.read()
	return

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("path_to_video", help="path to input video")
	parser.add_argument("output_path", help="filename or path where images will be saved. directory must already exist.")
	args = parser.parse_args()
	video_to_images(args.path_to_video, args.output_path)
	return

if __name__ == "__main__":
	main()
	print("done!")