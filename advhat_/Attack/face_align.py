import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../Demo/'))
import tensorflow as tf
import numpy as np
import cv2
import skimage.io as io
from skimage import transform as trans
from align import detect_face
from stn import spatial_transformer_network as stn
from utils import projector

# Align face as ArcFace template
def preprocess(img, landmark):
    image_size = [250,250]
    src = 250./112.*np.array([
		[38.2946, 51.6963],
		[73.5318, 51.5014],
		[56.0252, 71.7366],
		[41.5493, 92.3655],
		[70.7299, 92.2041] ], dtype=np.float32)
    dst = landmark.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]

    warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
    return warped

def main(args):
        sess = tf.Session()
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
        threshold = [ 0.6, 0.7, 0.7 ]
        factor = 0.709

        img = io.imread(args.image)
        #img = io.imread('/face/Mask/AdversarialMask/datasets/'+str(i).zfill(3)+'.jpg')

        _minsize = min(min(img.shape[0]//5, img.shape[1]//5),80)
        bounding_boxes, points = detect_face.detect_face(img, _minsize, pnet, rnet, onet, threshold, factor)
        assert bounding_boxes.size>0
        points = points[:, 0]
        landmark = points.reshape((2,5)).T
        warped = preprocess(img, landmark)

        #io.imsave('/face/Mask/AdversarialMask/datasets/'+str(i).zfill(3)+'_aligned.png',warped)
        io.imsave(args.out, warped)

	
def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('image', type=str, help='Path to the image.')
    parser.add_argument('out', type=str, help='Path to the image.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))


