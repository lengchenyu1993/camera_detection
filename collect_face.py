import cv2
import time
import sys
import os
import argparse
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 480)
video_capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('dir_path')
    parser.add_argument('person_name')

    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    dir_path = args.dir_path
    person_name = args.person_name
    person_dir = os.path.join(dir_path, person_name)
    if not os.path.exists(person_dir):
        os.mkdir(person_dir)
    start = time.time()
    run_time = 10
    c = 0
    timeF = 5
    while (time.time() - start < run_time):
        # Capture frame-by-frame
        c += 1
        c = c % timeF
        ret, frame = video_capture.read()
        cv2.imshow('frame', frame)
        if(c == 0): #save as jpg every 10 frame
             cv2.imwrite(os.path.join(person_dir, str(time.time()) + '.jpg'), frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture

    video_capture.release()
    cv2.destroyAllWindows()
