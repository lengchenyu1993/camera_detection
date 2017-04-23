import dlib
import cv2
import sys
import time
import argparse
import pickle
import logging
import numpy as np
import tensorflow as tf
from scipy import misc
from PIL import Image
from multiprocessing import Queue, Process, Manager, Event

def rectangles_to_array(rectangles):
    bbs = np.zeros([len(rectangles), 4])
    for (i, face) in enumerate(rectangles):
        bbs[i][0] = face.left()
        bbs[i][1] = face.top()
        bbs[i][2] = face.right()
        bbs[i][3] = face.bottom()
    return bbs

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

def process_img(img, image_size, detector):
    margin = 45
    img = img[..., ::-1]
    faces = detector(img, 0)
    face_num = len(faces)
    if face_num == 0:
        return None
    bounding_boxes = rectangles_to_array(faces)
    det = bounding_boxes[:,0:4]
    img_size = np.asarray(img.shape)[0:2]
    if face_num > 1:
        bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
        img_center = img_size / 2
        offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
        offset_dist_squared = np.sum(np.power(offsets,2.0),0)
        index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
        det = det[index,:]
    det = np.squeeze(det)
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0]-margin/2, 0)
    bb[1] = np.maximum(det[1]-margin/2, 0)
    bb[2] = np.minimum(det[2]+margin/2, img_size[1])
    bb[3] = np.minimum(det[3]+margin/2, img_size[0])
    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
    aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
    images = np.zeros((1, image_size, image_size, 3))
    img = prewhiten(aligned)
    images[0,:,:,:] = img
    return (int(det[0]), int(det[1]), int(det[2]), int(det[3]), images)



class TfProcess(Process):

    def __init__(self, face_queues, detect_person):
        Process.__init__(self)
        self.exit = Event()
        self.face_queues = face_queues
        self.detect_person = detect_person

    def run(self):
        GRAPH_PATH = './graph.pb'
        classfier = './classfier.txt'
        face_data = pickle.load(open('face_data.txt', 'r'))
        others = 'others'
        now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        log_file = now + '.log'
        logging.basicConfig(filename=log_file, level=logging.INFO)

        with open(classfier, 'r') as f:
            (le, clf) = pickle.load(f)

        with tf.gfile.FastGFile(GRAPH_PATH, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

        sess = tf.InteractiveSession()
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

        while not self.exit.is_set():
            if not self.face_queues.empty():
                image = self.face_queues.get()
                feed_dict = {images_placeholder:image,  phase_train_placeholder:False}
                embd_feature = sess.run(embeddings, feed_dict=feed_dict)
                feature = embd_feature[0]
                predictions = clf.predict_proba(feature.reshape(1, -1)).ravel()
                maxI = np.argmax(predictions)
                person = le.inverse_transform(maxI)
                confidence = predictions[maxI]
                if confidence < 0.8:
                    self.detect_person['who'] = 'others'
                else:
                    self.detect_person['who'] = person
                logging.info(predictions)
                logging.info(le.classes_)
                # time.sleep(0.01)

    def shutdown(self):
        print('stop!!!!!!!')
        self.exit.set()




if __name__ == '__main__':
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 360)
    video_capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 360)
    detector = dlib.get_frontal_face_detector()
    face_queues = Queue(2)
    manager = Manager()
    detect_person = manager.dict()
    detect_person['who'] = None
    p = TfProcess(face_queues, detect_person)
    p.start()
    while True:
        ret, frame = video_capture.read()
        result = process_img(frame, 160, detector)
        if result:
            left, top, right, bottom, image = result
            if not face_queues.full():
                face_queues.put(image)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255,0), 2)
        else:
            detect_person['who'] = None
        if detect_person['who']:
            print(detect_person['who'])
            cv2.putText(frame, detect_person['who'], (50,100),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0 ,0),
                thickness = 2, lineType = 2)
        cv2.imshow('Vedio', frame)
        # time.sleep(0.01)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    p.shutdown()
    video_capture.release()
    cv2.destroyAllWindows()
