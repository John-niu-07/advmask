import argparse
import sys
import tensorflow as tf
import numpy as np
import skimage.io as io
from skimage.transform import rescale
from time import time

# Prepare image to network input format
def prep(im):
    if len(im.shape)==3:
        return np.transpose(im,[2,0,1]).reshape((1,3,112,112))*2-1
    elif len(im.shape)==4:
        return np.transpose(im,[0,3,1,2]).reshape((im.shape[0],3,112,112))*2-1




class Infere():
    def __init__(self):
        #self.sess = tf.Session()

        with tf.gfile.GFile('/face/hat/advhat/r100.pb', "rb") as f:
                self.graph_def = tf.GraphDef()
                self.graph_def.ParseFromString(f.read())
        #        tf.import_graph_def(graph_def, input_map=None, return_elements=None, name="")

        #self.sess.run(tf.global_variables_initializer())

    def inf(self, im):
        with tf.Graph().as_default() as g_:
            tf.import_graph_def(self.graph_def, input_map=None, return_elements=None, name="")
            
        #sess = tf.Session(graph=g_)
        #sess.run(tf.global_variables_initializer())
        with tf.Session(graph=g_) as sess:
                  
            image_input = sess.graph.get_tensor_by_name('image_input:0')
            keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
            is_train = sess.graph.get_tensor_by_name('training_mode:0')
            #embedding = tf.get_default_graph().get_tensor_by_name('embedding:0')
            embedding = sess.graph.get_tensor_by_name('embedding:0')

            tfdict = {keep_prob:1.0, is_train:False}
            tfdict[image_input] = prep(im)
            emb = sess.run(embedding,feed_dict=tfdict)

        return emb
