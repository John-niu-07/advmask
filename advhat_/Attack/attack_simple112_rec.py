import argparse
import sys
import os
import tensorflow as tf
import numpy as np
import skimage.io as io
from skimage.transform import rescale
from tqdm import tqdm
from stn import spatial_transformer_network as stn
from utils import TVloss, projector
from sklearn.linear_model import LinearRegression as LR
from time import time
import datetime
import matplotlib.pyplot as plt

# Prepare image to network input format
def prep(im):
    if len(im.shape)==3:
        return np.transpose(im,[2,0,1]).reshape((1,3,112,112))*2-1
    elif len(im.shape)==4:
        return np.transpose(im,[0,3,1,2]).reshape((im.shape[0],3,112,112))*2-1

def main(args):
        print(args)
        now = str(datetime.datetime.now())
        
        sess = tf.Session()
        
        # Off-plane sticker projection
        logo = tf.placeholder(tf.float32,shape=[None,40,50,3],name='logo_input')
        face_input = tf.placeholder(tf.float32,shape=[None,112,112,3],name='face_input')
        #theta = tf.placeholder(tf.float32,shape=[None,6],name='theta_input')



        mask = np.zeros((1,112,112,3))
        mask[0, 10:50, 31:81, :] = np.ones((40, 50, 3))

        logo_img = tf.pad(logo, [[0,0],[10,62],[31,31],[0,0]])
        united = logo_img * mask + face_input * (1-mask)

        #hat = face_input * mask


        #final_crop = tf.clip_by_value(stn(united, theta, (112,112)), 0., 1.)
        #final_crop = tf.clip_by_value(stn(united, theta), 0., 1.)
        final_crop = tf.clip_by_value(united, 0., 1.)
        
        # TV loss and gradients
        w_tv = tf.placeholder(tf.float32,name='w_tv_input')
        tv_loss = TVloss(logo,w_tv)

        grads_tv = tf.gradients(tv_loss,logo)
        grads_input = tf.placeholder(tf.float32,shape=[None,112,112,3],name='grads_input')
        grads1 = tf.gradients(final_crop,logo,grad_ys=grads_input)
        
        # Varios images generator
        class Imgen(object):
                def __init__(self):
                        self.fdict = {w_tv:args.w_tv}
                        
                def gen_fixed(self,im,advhat):
                        #logo_img = np.zeros((600,600,3))
                        #logo_img[50:100, 200:400 :] = advhat
                        #self.fdict[logo] = np.expand_dims(advhat,0)
                        self.fdict[logo] = np.expand_dims(advhat,0)
                        self.fdict[face_input] = np.expand_dims(im,0)
                        return self.fdict, sess.run(final_crop,feed_dict=self.fdict)
                
                        
        gener = Imgen()

        

        fdict = {}



        im0 = io.imread(args.image)/255.
        #fdict_val, im_val = gener.gen_fixed(im0,init_logo)
        fdict[face_input] = np.expand_dims(im0,0)

        # Initialization of the sticker
        init_logo = np.ones((40,50,3))*127./255.


        init_logo[:] = io.imread(args.logo0)/255.
        fdict[logo] = np.expand_dims(init_logo,0)
        img_out = sess.run(final_crop,feed_dict=fdict)
        io.imsave('rec112_logo_0.png', img_out[0])
        #io.imsave('diff112_0.png', np.abs(im0 - img_out[0])*80)
        #old_img = img_out[0].copy()

        init_logo[:] = io.imread(args.logo1)/255.
        fdict[logo] = np.expand_dims(init_logo,0)
        img_out = sess.run(final_crop,feed_dict=fdict)
        io.imsave('rec112_logo_20.png', img_out[0])
        #io.imsave('diff112_1.png', np.abs(img_out[0]- old_img)*80)
        #old_img = img_out[0].copy()

        init_logo[:] = io.imread(args.logo2)/255.
        fdict[logo] = np.expand_dims(init_logo,0)
        img_out = sess.run(final_crop,feed_dict=fdict)
        io.imsave('rec112_logo_40.png', img_out[0])
        #io.imsave('diff112_2.png', np.abs(img_out[0]- old_img)*80)
        #old_img = img_out[0].copy()

        init_logo[:] = io.imread(args.logo3)/255.
        fdict[logo] = np.expand_dims(init_logo,0)
        img_out = sess.run(final_crop,feed_dict=fdict)
        io.imsave('rec112_logo_60.png', img_out[0])
        #io.imsave('diff112_3.png', np.abs(img_out[0]-old_img)*80)
        #old_img = img_out[0].copy()


        init_logo[:] = io.imread(args.logo4)/255.
        fdict[logo] = np.expand_dims(init_logo,0)
        img_out = sess.run(final_crop,feed_dict=fdict)
        io.imsave('rec112_logo_80.png', img_out[0])
        #io.imsave('diff112_4.png', np.abs(img_out[0]-old_img)*80)


        '''
        img_hat = sess.run(hat,feed_dict=fdict)
        #print(img_hat.shape)
        hat_out = np.zeros((50,200,3))
        hat_out = img_hat[0, 50:100, 200:400, : ]
        io.imsave('hat_out.png', hat_out)
        '''

        '''
        print(over)




        # Embedding model
        with tf.gfile.GFile(args.model, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def,
                                          input_map=None,
                                          return_elements=None,
                                          name="")
        sess.run(tf.global_variables_initializer())

        image_input = tf.get_default_graph().get_tensor_by_name('image_input:0')
        keep_prob = tf.get_default_graph().get_tensor_by_name('keep_prob:0')
        is_train = tf.get_default_graph().get_tensor_by_name('training_mode:0')
        embedding = tf.get_default_graph().get_tensor_by_name('embedding:0')

        orig_emb = tf.placeholder(tf.float32,shape=[None,512],name='orig_emb_input')
        cos_loss = tf.reduce_sum(tf.multiply(embedding,orig_emb),axis=1)
        #cos_loss = tf.abs(tf.reduce_sum(tf.multiply(embedding,orig_emb),axis=1))
        #cos_loss = 1 - tf.reduce_sum(tf.multiply(embedding,orig_emb),axis=1)
        grads2 = tf.gradients(cos_loss,image_input)

        fdict2 = {keep_prob:1.0,is_train:False}
        
        # Anchor embedding calculation
        if args.anchor_face!=None:
                anch_im = rescale(io.imread(args.anchor_face)/255.,(112./600., 112./600., 1),order=5)
                fdict2[image_input] = prep(anch_im)
                fdict2[orig_emb] = sess.run(embedding,feed_dict=fdict2)
        elif args.anchor_emb!=None:
                fdict2[orig_emb] = np.load(args.anchor_emb)[-1:]
        else:
                anch_im = rescale(io.imread(args.image)/255.,(112./600., 112./600., 1),order=5)
                fdict2[image_input] = prep(anch_im)
                fdict2[orig_emb] = sess.run(embedding,feed_dict=fdict2)
        
        # Attack constants
        im0 = io.imread(args.image)/255.
        regr = LR(n_jobs=4)
        regr_len = 100
        regr_coef = -1.
        moments = np.zeros((50,200,3))
        moment_val = 0.9
        #step_val = 1./51.
        #step_val = 1./255.
        step_val = 0.00001
        stage = 1
        step = 0
        #lr_thresh = 10
        lr_thresh = 100
        ls = []
        t = time()
        while True:
                # Projecting sticker to the face and feeding it to the embedding model
                #fdict,ims = gener.gen_random(im0,init_logo)
                fdict,ims = gener.gen_fixed(im0,init_logo)
                fdict2[image_input] = prep(ims)
                grad_tmp = sess.run(grads2,feed_dict=fdict2)
                #print(np.transpose(grad_tmp[0],[0,2,3,1])[0])
                
                fdict_val, im_val = gener.gen_fixed(im0,init_logo)
                fdict2[image_input] = prep(im_val)
                ls.append(sess.run(cos_loss,feed_dict=fdict2)[0])

                
                #emb1 = sess.run(embedding,feed_dict=fdict2)
                #anch_im = rescale(io.imread(args.image)/255.,(112./600., 112./600., 1),order=5)
                #fdict2[image_input] = prep(anch_im)
                #emb2 = sess.run(embedding,feed_dict=fdict2)
                #cos_sim = np.sum(emb1 * emb2)
                #print('Cos_sim((emb1, orig_emb) =', cos_sim)
                

                
                #io.imsave('logo_'+str(step)+'.png',init_logo)
                #io.imsave('tmp_'+str(step)+'.png',prep(im_val))
                #print(ls)

                # Gradients to the original sticker image
                fdict[grads_input] = np.transpose(grad_tmp[0],[0,2,3,1])
                grads_on_logo = np.mean(sess.run(grads1,feed_dict=fdict)[0],0)
                #print(grads_on_logo[0])
                #grads_on_logo += sess.run(grads_tv,feed_dict=fdict)[0][0]
                #print(grads_on_logo[0][0])
                moments = moments*moment_val + grads_on_logo*(1.-moment_val)
                #print(init_logo[0])
                init_logo -= step_val*np.sign(moments)
                #init_logo += step_val*np.sign(moments)
                init_logo = np.clip(init_logo,0.,1.)
                
                # Logging
                if step%20==0:
                        print('Stage:',stage,'Step:',step,'Loss:',round(ls[-1],5))
                        io.imsave('logo_'+str(step)+'.png',init_logo)
                        #io.imsave('tmp_'+str(step)+'.png',prep(im_val))
                        #print(ls)
                step += 1

                # Switching to the second stage
                if step>lr_thresh:
                        regr.fit(np.expand_dims(np.arange(100),1),np.hstack(ls[-100:]))
                        regr_coef = regr.coef_[0]
                        print(regr_coef)
                        if regr_coef>=0:
                                if stage==1:
                                        stage = 2
                                        moment_val = 0.995
                                        #step_val = 1./255.
                                        step_val = 1./2550.
                                        step = 0
                                        regr_coef = -1.
                                        lr_thresh = 200
                                        t = time()
                                else:
                                        break

        print(ls)

        plt.plot(range(len(ls)),ls)
        #plt.savefig(now+'_cosine.png')
        plt.savefig('cosine.png')
        #io.imsave(now+'_advhat.png',init_logo)
        io.imsave('advhat.png',init_logo)
        '''


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('image', type=str, help='Path to the image for attack.')
    parser.add_argument('model', type=str, help='Path to the model for attack.')
    parser.add_argument('--init_face', type=str, default=None, help='Path to the face for sticker inititalization.')
    parser.add_argument('--logo0', type=str, default=None, help='Path to the image for inititalization.')
    parser.add_argument('--logo1', type=str, default=None, help='Path to the image for inititalization.')
    parser.add_argument('--logo2', type=str, default=None, help='Path to the image for inititalization.')
    parser.add_argument('--logo3', type=str, default=None, help='Path to the image for inititalization.')
    parser.add_argument('--logo4', type=str, default=None, help='Path to the image for inititalization.')
    parser.add_argument('--anchor_face', type=str, default=None, help='Path to the anchor face.')
    parser.add_argument('--anchor_emb', type=str, default=None, help='Path to the anchor emb (the last will be used)')
    parser.add_argument('--w_tv', type=float, default=1e-4, help='Weight of the TV loss')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for attack')
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
