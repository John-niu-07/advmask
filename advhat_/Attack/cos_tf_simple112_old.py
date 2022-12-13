import argparse
import sys
import tensorflow as tf
import numpy as np
import skimage.io as io
from skimage.transform import rescale
from stn import spatial_transformer_network as stn

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
        theta = tf.placeholder(tf.float32,shape=[None,6],name='theta_input')

        tfdict = {keep_prob:1.0, is_train:False}

        '''
        face_input = tf.placeholder(tf.float32,shape=[None,600,600,3],name='face_input')
        theta = tf.placeholder(tf.float32,shape=[None,6],name='theta_input')
        final_crop_orig = tf.clip_by_value(stn(face_input, theta, (112,112)), 0., 1.)
        
        tfdict[image_input] = im2
        anch_im = io.imread(args.face1)/255.
        fdict = {theta:[[1.,0.,0.,0.,1.,0.]]}
        fdict[face_input] = np.expand_dims(anch_im,0)
        tfdict[image_input] = prep(sess.run(final_crop_orig, feed_dict=fdict))
        emb1 = sess.run(embedding,feed_dict=tfdict)
        '''



        # Embedding calculation
        ##im1 = prep(rescale(io.imread(args.face1)/255.,112./600.,order=5))
        #im1 = prep(io.imread(args.face1)/255.)
        #np.save('im1_np', im1)
        im1 = np.load('im1_np.npy')
        #print(im1[0,0:5])


        im1 = np.random.rand(1, 3, 112, 112)
        print(im1[0,0,1:9,0])
        tfdict[image_input] = im1
        emb1 = sess.run(embedding,feed_dict=tfdict)
        print(emb1[0,0:5])
        np.save('emb1_np', emb1)


        '''        
        im2 = prep(io.imread(args.face2)/255.)
        im3 = prep(io.imread(args.face3)/255.)
        im4 = prep(io.imread(args.face4)/255.)
        im5 = prep(io.imread(args.face5)/255.)
        np.save('im2_np', im2)
        np.save('im3_np', im3)
        np.save('im4_np', im4)
        np.save('im5_np', im5)

        
        im2 = np.load('im2_np.npy')
        im3 = np.load('im3_np.npy')
        im4 = np.load('im4_np.npy')
        im5 = np.load('im5_np.npy')
        '''
        im5 = np.random.rand(1, 3, 112, 112)
        im4 = np.random.rand(1, 3, 112, 112)
        im3 = np.random.rand(1, 3, 112, 112)
        im2 = np.random.rand(1, 3, 112, 112)
        

        print(im2[0,0,1:9,0])
        tfdict[image_input] = im2
        emb2 = sess.run(embedding,feed_dict=tfdict)
        print(emb2[0,0:5])
        np.save('emb2_np', emb2)
        print(im3[0,0,1:9,0])
        tfdict[image_input] = im3
        emb3 = sess.run(embedding,feed_dict=tfdict)
        print(emb3[0,0:5])
        np.save('emb3_np', emb3)
        print(im4[0,0,1:9,0])
        tfdict[image_input] = im4
        emb4 = sess.run(embedding,feed_dict=tfdict)
        print(emb4[0,0:5])
        np.save('emb4_np', emb4)
        print(im5[0,0,1:9,0])
        tfdict[image_input] = im5
        emb5 = sess.run(embedding,feed_dict=tfdict)
        print(emb5[0,0:5])
        np.save('emb5_np', emb5)


        # Result
        cos_sim12 = np.sum(emb1 * emb2)
        cos_sim13 = np.sum(emb1 * emb3)
        cos_sim14 = np.sum(emb1 * emb4)
        cos_sim15 = np.sum(emb1 * emb5)
        print('Cos_sim(1, 2) =', cos_sim12) 
        print('Cos_sim(1, 3) =', cos_sim13) 
        print('Cos_sim(1, 4) =', cos_sim14) 
        print('Cos_sim(1, 5) =', cos_sim15)


        '''
        cos_sim23 = np.sum(emb2 * emb3)
        cos_sim24 = np.sum(emb2 * emb4)
        cos_sim25 = np.sum(emb2 * emb5)

        cos_sim34 = np.sum(emb3 * emb4)
        cos_sim35 = np.sum(emb3 * emb5)

        cos_sim45 = np.sum(emb4 * emb5)
        
        print('Cos_sim(2, 3) =', cos_sim23) 
        print('Cos_sim(2, 4) =', cos_sim24) 
        print('Cos_sim(2, 5) =', cos_sim25)

        print('Cos_sim(3, 4) =', cos_sim34) 
        print('Cos_sim(3, 5) =', cos_sim35)

        print('Cos_sim(4, 5) =', cos_sim45) 
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
    
    parser.add_argument('--face1', type=str, help='Path to the preprocessed face1.')
    parser.add_argument('--face2', type=str, help='Path to the preprocessed face2.')
    parser.add_argument('--face3', type=str, help='Path to the preprocessed face3.')
    parser.add_argument('--face4', type=str, help='Path to the preprocessed face4.')
    parser.add_argument('--face5', type=str, help='Path to the preprocessed face5.')
    parser.add_argument('--model', type=str, help='Path to the model.')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
