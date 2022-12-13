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
        ''' 
        sess = tf.Session()
        
        # Embedding model
        with tf.gfile.GFile(args.model, "rb") as f:
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
        '''

        hat = io.imread('/face/hat/advhat/Attack/john_hat_aligned112.png')/255.
        hat_init = hat + np.random.rand(112, 112, 3)*0.2
        hat_init = np.clip(hat_init, 0., 1.)
        print('diff: %s' %(np.mean(np.abs( hat_init - hat))))
        io.imsave('/face/hat/advhat/Attack/logo/john_init.png', hat_init)

        hat = io.imread('/face/hat/advhat/Attack/hat_out112.png')/255.
        hat_init = hat + np.random.rand(40, 50, 3)*0.2
        hat_init = np.clip(hat_init, 0., 1.)
        print('diff: %s' %(np.mean(np.abs( hat_init - hat))))
        io.imsave('/face/hat/advhat/Attack/logo/hat_init.png', hat_init)
        


        # Embedding calculation
        target = io.imread('/face/hat/advhat/Attack/hat_out112.png')/255.

        logo_0  = io.imread('/face/hat/advhat/Attack/logo/logo112_0.png')/255.
        logo_20 = io.imread('/face/hat/advhat/Attack/logo/logo112_20.png')/255.
        logo_40 = io.imread('/face/hat/advhat/Attack/logo/logo112_40.png')/255.
        logo_60 = io.imread('/face/hat/advhat/Attack/logo/logo112_60.png')/255.
        logo_80 = io.imread('/face/hat/advhat/Attack/logo/logo112_80.png')/255.


        diff_0 = target - logo_0 
        diff_1 = target - logo_20 
        diff_2 = target - logo_40
        diff_3 = target - logo_60
        diff_4 = target - logo_80

        #diff_01 = logo_0 - logo_20 
        #diff_12 = logo_20 - logo_40

        print('diff_0: %s' %(np.mean(np.abs(diff_0))))
        print('diff_1: %s' %(np.mean(np.abs(diff_1))))
        print('diff_2: %s' %(np.mean(np.abs(diff_2))))
        print('diff_3: %s' %(np.mean(np.abs(diff_3))))
        print('diff_4: %s' %(np.mean(np.abs(diff_4))))

        #print('diff_01: %s' %(np.mean(np.abs(diff_01))))
        #print('diff_12: %s' %(np.mean(np.abs(diff_12))))
       
        '''
        im_target = io.imread('/face/hat/advhat/Attack/john_hat_aligned112.png')/255.

        im_0  = io.imread('/face/hat/advhat/Attack/rec112_logo0.png')/255.
        im_20 = io.imread('/face/hat/advhat/Attack/rec112_logo1.png')/255.
        im_40 = io.imread('/face/hat/advhat/Attack/rec112_logo2.png')/255.
        im_60 = io.imread('/face/hat/advhat/Attack/rec112_logo3.png')/255.

        tfdict[image_input] = prep(im_target)
        emb_target = sess.run(embedding,feed_dict=tfdict)
        np.save('emb_target', emb_target)


        tfdict[image_input] = prep(im_0)
        emb_0 = sess.run(embedding,feed_dict=tfdict)
        np.save('emb_0', emb_0)

        tfdict[image_input] = prep(im_20)
        emb_20 = sess.run(embedding,feed_dict=tfdict)
        np.save('emb_20', emb_20)
        

        tfdict[image_input] = prep(im_40)
        emb_40 = sess.run(embedding,feed_dict=tfdict)
        np.save('emb_40', emb_40)


        tfdict[image_input] = prep(im_60)
        emb_60 = sess.run(embedding,feed_dict=tfdict)
        np.save('emb_60', emb_60)


        # Result
        cos_sim_0 = np.sum(emb_target * emb_0)
        cos_sim_20 = np.sum(emb_target * emb_20)
        cos_sim_40 = np.sum(emb_target * emb_40)
        cos_sim_60 = np.sum(emb_target * emb_60)
        #cos_sim_80 = np.sum(emb_target * emb_80)
        print('Cos_sim(target, 0) =', cos_sim_0) 
        print('Cos_sim(target, 20) =', cos_sim_20) 
        print('Cos_sim(target, 40) =', cos_sim_40) 
        print('Cos_sim(target, 60) =', cos_sim_60)
        #print('Cos_sim(target, 80) =', cos_sim_80)


        
        cos_sim0_20 = np.sum(emb_0 * emb_20)
        cos_sim20_40 = np.sum(emb_20 * emb_40)
        cos_sim0_40 = np.sum(emb_0 * emb_40)

        #cos_sim34 = np.sum(emb3 * emb4)
        #cos_sim35 = np.sum(emb3 * emb5)

        #cos_sim45 = np.sum(emb4 * emb5)
        
        print('Cos_sim(0, 20) =', cos_sim0_20) 
        print('Cos_sim(20, 40) =', cos_sim20_40) 
        print('Cos_sim(0, 40) =', cos_sim0_40)
        '''
def parse_arguments(argv):

    '''   
    #hat = io.imread('/art/adversarial-robustness-toolbox/examples/dataset/john_hat_hatimg.png')
    hat = io.imread('/art/adversarial-robustness-toolbox/examples/dataset/john_hat_hatimg2.png')
    print(hat.shape)
    #hat_crop = hat[160:440, 26:574, :]
    hat_crop = hat[60:514, 26:574, :]
    print(hat_crop.shape)
    #hat_crop = rescale(hat_crop, (400./454., 900./548., 1), order=5)
    hat_crop = rescale(hat_crop, (900./454., 900./548., 1), order=5)
    #io.imsave('john_hatcrop.png',hat_crop)
    io.imsave('john_hatcrop2.png',hat_crop)
    '''
    
    parser = argparse.ArgumentParser()
    ''' 
    parser.add_argument('--face1', type=str, help='Path to the preprocessed face1.')
    parser.add_argument('--face2', type=str, help='Path to the preprocessed face2.')
    parser.add_argument('--face3', type=str, help='Path to the preprocessed face3.')
    parser.add_argument('--face4', type=str, help='Path to the preprocessed face4.')
    parser.add_argument('--face5', type=str, help='Path to the preprocessed face5.')
    
    parser.add_argument('--model', type=str, help='Path to the model.')
    '''
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
