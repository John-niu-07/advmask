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
    image_size = [600,600]
    src = 600./112.*np.array([
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
	_minsize = min(min(img.shape[0]//5, img.shape[1]//5),80)
	bounding_boxes, points = detect_face.detect_face(img, _minsize, pnet, rnet, onet, threshold, factor)
	assert bounding_boxes.size>0
	points = points[:, 0]
	landmark = points.reshape((2,5)).T
	warped = preprocess(img, landmark)

	#io.imsave(args.image[:-4]+'_aligned.png',warped)

	if args.mask:
		logo_mask = np.ones((1,400,900,3),dtype=np.float32)

		logo = tf.placeholder(tf.float32,shape=[1,400,900,3])
		param = tf.placeholder(tf.float32,shape=[1,1])
		ph = tf.placeholder(tf.float32,shape=[1,1])
		result = projector(param,ph,logo)

		face_input = tf.placeholder(tf.float32,shape=[1,600,600,3])
		theta = tf.placeholder(tf.float32,shape=[1,6])
		prepared = stn(result,theta)





            
		'''
		united = prepared[:,300:,150:750]+face_input*(1-prepared[:,300:,150:750])
		img_with_mask = sess.run(united,feed_dict={ph:[[args.ph]],logo:logo_mask,param:[[args.param]],\
										face_input:np.expand_dims(warped/255.,0),\
										theta:1./args.scale*np.array([[1.,0.,-args.x/450.,0.,1.,-args.y/450.]])})[0]

		
		#logo_input = (io.imread('/face/hat/advhat/Attack/922727_advhat.png')/255.).reshape((1,400,900,3))
		#logo_input = (io.imread('/face/hat/advhat/Attack/example.png')/255.).reshape((1,400,900,3))
		#logo_input = (io.imread('/face/hat/advhat/Attack/advhat.png')/255.).reshape((1,400,900,3))
		#logo_input = (io.imread('/face/hat/advhat/Attack/advhat_init.png')/255.).reshape((1,400,900,3))
		#logo_input = (io.imread('/face/hat/advhat/Attack/john_hatcrop.png')/255.).reshape((1,400,900,3))
		logo_input = (io.imread('/face/hat/advhat/Attack/logo_0.png')/255.).reshape((1,400,900,3))
		logo_input_2 = (io.imread('/face/hat/advhat/Attack/logo_24.png')/255.).reshape((1,400,900,3))
		logo_input_3 = (io.imread('/face/hat/advhat/Attack/logo_49.png')/255.).reshape((1,400,900,3))
		logo_input_4 = (io.imread('/face/hat/advhat/Attack/logo_99.png')/255.).reshape((1,400,900,3))
		logo2 = tf.placeholder(tf.float32,shape=[1,400,900,3])
		result2 = projector(param,ph,logo2)
		prepared2 = stn(result2,theta)

		united = prepared2[:,300:,150:750] * prepared[:,300:,150:750]+face_input*(1-prepared[:,300:,150:750])
		img_with_hat = sess.run(united,feed_dict={ph:[[args.ph]],logo:logo_mask,param:[[args.param]],logo2:logo_input,\
										face_input:np.expand_dims(warped/255.,0),\
										theta:1./args.scale*np.array([[1.,0.,-args.x/450.,0.,1.,-args.y/450.]])})[0]
		img_with_hat_2 = sess.run(united,feed_dict={ph:[[args.ph]],logo:logo_mask,param:[[args.param]],logo2:logo_input_2,\
										face_input:np.expand_dims(warped/255.,0),\
										theta:1./args.scale*np.array([[1.,0.,-args.x/450.,0.,1.,-args.y/450.]])})[0]
		img_with_hat_3 = sess.run(united,feed_dict={ph:[[args.ph]],logo:logo_mask,param:[[args.param]],logo2:logo_input_3,\
										face_input:np.expand_dims(warped/255.,0),\
										theta:1./args.scale*np.array([[1.,0.,-args.x/450.,0.,1.,-args.y/450.]])})[0]
		img_with_hat_4 = sess.run(united,feed_dict={ph:[[args.ph]],logo:logo_mask,param:[[args.param]],logo2:logo_input_4,\
										face_input:np.expand_dims(warped/255.,0),\
										theta:1./args.scale*np.array([[1.,0.,-args.x/450.,0.,1.,-args.y/450.]])})[0]

		'''
		
		logo_input = (io.imread('/face/hat/advhat/Attack/logo_0.png')/255.).reshape((1,400,900,3))
		logo2 = tf.placeholder(tf.float32,shape=[1,400,900,3])
		result2 = projector(param,ph,logo2)
		prepared2 = stn(result2,theta)

		united = prepared2[:,300:,150:750] * prepared[:,300:,150:750]+face_input*(1-prepared[:,300:,150:750])
		img_rec = sess.run(united,feed_dict={ph:[[args.ph]],logo:logo_mask,param:[[args.param]],logo2:logo_input,\
										face_input:np.expand_dims(warped/255.,0),\
										theta:1./args.scale*np.array([[1.,0.,-args.x/450.,0.,1.,-args.y/450.]])})[0]
		'''

		hat_crop = face_input * prepared[:,300:,150:750]
		img_hat_crop = sess.run(hat_crop,feed_dict={ph:[[args.ph]],logo:logo_mask,param:[[args.param]],\
										face_input:np.expand_dims(warped/255.,0),\
										theta:1./args.scale*np.array([[1.,0.,-args.x/450.,0.,1.,-args.y/450.]])})[0]

		#io.imsave(args.image[:-4]+'_mask.png',img_with_mask)
		#io.imsave(args.image[:-4]+'_hatimg.png',img_hat_crop)
		#io.imsave(args.image[:-4]+'_withhat.png',img_with_hat)
		io.imsave(args.image[:-4]+'_withlogo1.png',img_with_hat)
		io.imsave(args.image[:-4]+'_withlogo2.png',img_with_hat_2)
		io.imsave(args.image[:-4]+'_withlogo3.png',img_with_hat_3)
		io.imsave(args.image[:-4]+'_withlogo4.png',img_with_hat_4)
	
		'''
		io.imsave(args.image[:-4]+'_rec.png',img_rec)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('image', type=str, help='Path to the image.')
    parser.add_argument('--mask', action='store_true', help='Use when search the sticker parameters')
    #parser.add_argument('--ph', type=float, default=17., help='Angle of the off-plane rotation')
    parser.add_argument('--ph', type=float, default=0., help='Angle of the off-plane rotation')
    parser.add_argument('--param', type=float, default=0.0013, help='Parabola rate for the off-plane parabolic transformation')
    #parser.add_argument('--scale', type=float, default=0.465, help='Scaling parameter for the sticker')
    #parser.add_argument('--scale', type=float, default=0.5, help='Scaling parameter for the sticker')
    parser.add_argument('--scale', type=float, default=0.7, help='Scaling parameter for the sticker')
    parser.add_argument('--x', type=float, default=0., help='Translation of the sticker along x-axis')
    #parser.add_argument('--y', type=float, default=-15., help='Translation of the sticker along y-axis')
    #parser.add_argument('--y', type=float, default=0., help='Translation of the sticker along y-axis')
    parser.add_argument('--y', type=float, default=150., help='Translation of the sticker along y-axis')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
