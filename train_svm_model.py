# encoding:utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from scipy import misc
import pdb
import time
# from matplotlib import pyplot as plt
import align.detect_face
import os
import argparse
import cv2
import pickle
import dlib
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# change the path
GRAPH_PATH = './graph.pb'

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

def process_img(img, image_size, detector, output_dir, person_name):
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
    img2 = Image.fromarray(aligned, 'RGB')
    # img1 = Image.fromarray(img, "RGB")
    # img1.save('1.jpeg')
    person_dir = os.path.join(output_dir, person_name)
    if not os.path.exists(person_dir):
        os.mkdir(person_dir)
    img2.save(os.path.join(person_dir, str(time.time()) + '.jpg'))
    return (int(det[0]), int(det[1]), int(det[2]), int(det[3]), images)





with tf.gfile.FastGFile(GRAPH_PATH, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

sess = tf.InteractiveSession()

detector = dlib.get_frontal_face_detector()

images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

if __name__=='__main__':
    clf_file = 'classfier.txt'
    clf = SVC(C=1, kernel='rbf', probability=True, gamma=2)
    postfix = ['.png', '.jpg']
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir')
    parser.add_argument('output_dir')
    args = parser.parse_args()
    img_dir = args.img_dir
    output_dir = args.output_dir
    dir_list = os.listdir(img_dir)
    for person_name in dir_list:
        person_dir = os.path.join(img_dir, person_name)
        face_data = os.path.join(output_dir, person_name + '.txt')
        if os.path.exists(face_data):
            continue
        if os.path.isdir(person_dir):
            imgs = []
            for img in os.listdir(person_dir):
                if os.path.splitext(img)[1] not in postfix:
                    continue
                image = cv2.imread(os.path.join(person_dir, img))
                img_crop = process_img(image, 160, detector, output_dir, person_name)
                if img_crop is not None:
                    imgs.append(img_crop[4])
            nrof_samples = len(imgs)
            images = np.zeros((nrof_samples, 160, 160, 3))
            for i in xrange(nrof_samples):
                images[i,:,:,:] = imgs[i][0]
            feed_dict = {images_placeholder:images,  phase_train_placeholder:False}
            embd_feature = sess.run(embeddings, feed_dict=feed_dict)
            with open(face_data, 'w') as f:
                pickle.dump(embd_feature, f)

    persons_lists = []
    labels =[]
    for item in os.listdir(output_dir):
        if item.endswith('.txt'):
            person = pickle.load(open(os.path.join(output_dir, item), 'r'))
            persons_lists.append(person)
            persons_name = item.split('.')[0]
            for i in xrange(len(person)):
                labels.append(persons_name)

    persons_feature = np.vstack(tuple(persons_lists))
    le = LabelEncoder()
    le.fit(labels)
    clf.fit(persons_feature, le.transform(labels))

    with open(clf_file, 'w') as f:
        pickle.dump((le, clf), f)




# img_paths = ['/home/wenfahu/facenet/data/images/Anthony_Hopkins_0001.jpg',
# '/home/wenfahu/facenet/data/images/Anthony_Hopkins_0002.jpg',
# '/home/wenfahu/faces/facenet/data/images/Woody_Allen_0001.png',
# '/home/wenfahu/faces/facenet/data/images/Wally_Szczerbiak_0001.png']
