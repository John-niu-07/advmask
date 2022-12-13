import argparse
import sys
import tensorflow as tf
import numpy as np
import skimage.io as io
from skimage.transform import rescale

# Prepare image to network input format
def prep(im):
    if len(im.shape)==3:
        return np.transpose(im,[2,0,1]).reshape((1,3,112,112))*2-1
    elif len(im.shape)==4:
        return np.transpose(im,[0,3,1,2]).reshape((im.shape[0],3,112,112))*2-1

def main(args):
        print(args)
         
        sess = tf.Session()
        
        # Embedding model
        with tf.gfile.GFile('/face/hat/advhat/r100.pb', "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

        sess.graph.as_default()

        tf.import_graph_def(graph_def,
                                          input_map=None,
                                          return_elements=None,
                                          name="")

        opname = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        #print(opname)


        #sess.run(tf.global_variables_initializer())



        image_input = tf.get_default_graph().get_tensor_by_name('image_input:0')
        keep_prob = tf.get_default_graph().get_tensor_by_name('keep_prob:0')
        is_train = tf.get_default_graph().get_tensor_by_name('training_mode:0')
        embedding = tf.get_default_graph().get_tensor_by_name('embedding:0')

        tfdict = {keep_prob:1.0, is_train:False}
        

        #hat = io.imread('/face/hat/advhat/Attack/hat_out112.png')/255.
        #hat_init = hat + np.random.rand(40, 50, 3)*0.2
        #print('diff: %s' %(np.mean(np.abs( hat_init - hat))))
        #io.imsave('/face/hat/advhat/Attack/logo/hat_init.png', hat_init)
        


        im_target = io.imread('/face/hat/advhat/Attack/john_hat_aligned112.png')/255.

        im_0  = io.imread('/face/hat/advhat/Attack/rec112_full_0.png')/255.
        im_20 = io.imread('/face/hat/advhat/Attack/rec112_full_20.png')/255.
        im_40 = io.imread('/face/hat/advhat/Attack/rec112_full_40.png')/255.
        im_60 = io.imread('/face/hat/advhat/Attack/rec112_full_60.png')/255.
        im_80 = io.imread('/face/hat/advhat/Attack/rec112_full_80.png')/255.

        diff_0 = im_target - im_0 
        diff_1 = im_target - im_20 
        diff_2 = im_target - im_40
        diff_3 = im_target - im_60
        diff_4 = im_target - im_80


        print('diff_0: %s' %(np.mean(np.abs(diff_0))))
        print('diff_1: %s' %(np.mean(np.abs(diff_1))))
        print('diff_2: %s' %(np.mean(np.abs(diff_2))))
        print('diff_3: %s' %(np.mean(np.abs(diff_3))))
        print('diff_4: %s' %(np.mean(np.abs(diff_4))))

       
        

        tfdict[image_input] = prep(im_target)
        emb_target = sess.run(embedding,feed_dict=tfdict)
        #np.save('emb_target', emb_target)


        tfdict[image_input] = prep(im_0)
        emb_0 = sess.run(embedding,feed_dict=tfdict)
        #np.save('emb_0', emb_0)

        tfdict[image_input] = prep(im_20)
        emb_20 = sess.run(embedding,feed_dict=tfdict)
        #np.save('emb_20', emb_20)
        

        tfdict[image_input] = prep(im_40)
        emb_40 = sess.run(embedding,feed_dict=tfdict)
        #np.save('emb_40', emb_40)


        tfdict[image_input] = prep(im_60)
        emb_60 = sess.run(embedding,feed_dict=tfdict)
        #np.save('emb_60', emb_60)

        tfdict[image_input] = prep(im_80)
        emb_80 = sess.run(embedding,feed_dict=tfdict)
        #np.save('emb_80', emb_80)

        # Result
        cos_sim_0 = np.sum(emb_target * emb_0)
        cos_sim_20 = np.sum(emb_target * emb_20)
        cos_sim_40 = np.sum(emb_target * emb_40)
        cos_sim_60 = np.sum(emb_target * emb_60)
        cos_sim_80 = np.sum(emb_target * emb_80)
        print('Cos_sim(target, 0) =', cos_sim_0) 
        print('Cos_sim(target, 20) =', cos_sim_20) 
        print('Cos_sim(target, 40) =', cos_sim_40) 
        print('Cos_sim(target, 60) =', cos_sim_60)
        print('Cos_sim(target, 80) =', cos_sim_80)


        
        cos_sim0_20 = np.sum(emb_0 * emb_20)
        cos_sim20_40 = np.sum(emb_20 * emb_40)
        cos_sim0_40 = np.sum(emb_0 * emb_40)

        #cos_sim34 = np.sum(emb3 * emb4)
        #cos_sim35 = np.sum(emb3 * emb5)

        #cos_sim45 = np.sum(emb4 * emb5)
        print('-----')
        print('Cos_sim(0, 20) =', cos_sim0_20) 
        print('Cos_sim(20, 40) =', cos_sim20_40) 
        print('Cos_sim(0, 40) =', cos_sim0_40)
        
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
