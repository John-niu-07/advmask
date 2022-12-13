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




def Infere(im):



    with tf.gfile.GFile('/face/hat/advhat/r100.pb', "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as g_:
            tf.import_graph_def(graph_def, input_map=None, return_elements=None, name="")

    with tf.Session(graph=g_) as sess:

            image_input = sess.graph.get_tensor_by_name('image_input:0')
            keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
            is_train = sess.graph.get_tensor_by_name('training_mode:0')
            #embedding = tf.get_default_graph().get_tensor_by_name('embedding:0')
            embedding = sess.graph.get_tensor_by_name('embedding:0')

            sess.run(tf.global_variables_initializer())
            tfdict = {keep_prob:1.0, is_train:False}
            tfdict[image_input] = prep(im)
            emb = sess.run(embedding,feed_dict=tfdict)

            print(emb[0,0:5])
            #print('Av. time:',round((time()-t),2))




    return emb


def main(args):
        print(args)
        t = time()



        im_target = io.imread('/face/hat/advhat/Attack/john_hat_aligned112.png')/255.

        im_0  = io.imread('/face/hat/advhat/Attack/rec112_logo_0.png')/255.
        im_20 = io.imread('/face/hat/advhat/Attack/rec112_logo_20.png')/255.
        im_40 = io.imread('/face/hat/advhat/Attack/rec112_logo_40.png')/255.
        im_60 = io.imread('/face/hat/advhat/Attack/rec112_logo_60.png')/255.
        im_80 = io.imread('/face/hat/advhat/Attack/rec112_logo_80.png')/255.




        '''
        
        # Embedding model
        with tf.gfile.GFile('/face/hat/advhat/r100.pb', "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, input_map=None, return_elements=None, name="")



        sess = tf.Session()
        #sess.graph.as_default()
        sess.run(tf.global_variables_initializer())


        image_input = sess.graph.get_tensor_by_name('image_input:0')
        keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
        is_train = sess.graph.get_tensor_by_name('training_mode:0')
        embedding = tf.get_default_graph().get_tensor_by_name('embedding:0')




        tfdict = {keep_prob:1.0, is_train:False}
        
        '''
        Inf  = Infere()
        emb_target = Inf.inf(im_target)
        print(emb_target[0,0:5])
        print('Av. time:',round((time()-t),2))
        #np.save('emb_target', emb_target)


        emb_0 = Inf.inf(im_0)
        print(emb_0[0,0:5])
        print('Av. time:',round((time()-t),2))
 

    
        '''
        tfdict[image_input] = prep(im_target)
        
        emb_target = sess.run(embedding,feed_dict=tfdict)
        print(emb_target[0,0:5])
        np.save('emb_target', emb_target)
        
        


        tfdict[image_input] = prep(im_0)
        emb_0 = sess.run(embedding,feed_dict=tfdict)
        print(emb_0[0,0:5])
        np.save('emb_0', emb_0)
        
        

        tfdict[image_input] = prep(im_20)
        emb_20 = sess.run(embedding,feed_dict=tfdict)
        print(emb_20[0,0:5])
        np.save('emb_20', emb_20)
        '''

        

        # Result
        cos_sim_0 = np.sum(emb_target * emb_0)
        
        cos_sim_20 = np.sum(emb_target * emb_20)
        
        print('Cos_sim(target, 0) =', cos_sim_0) 
        
        print('Cos_sim(target, 20) =', cos_sim_20) 
        

       

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
