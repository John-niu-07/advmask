"""
The script demonstrates a simple example of using ART with TensorFlow v1.x. The example train a small model on the MNIST
dataset and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train
the model, it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

#from art.attacks.evasion import FastGradientMethod
#from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
#from art.attacks.evasion.hop_skip_jump import HopSkipJump
#from art.attacks.evasion.geometric_decision_based_attack import GeoDA

from art.attacks.evasion.universal_perturbation import UniversalPerturbation
from art.attacks.evasion.targeted_universal_perturbation import TargetedUniversalPerturbation


from art.estimators.classification import TensorFlowV2Classifier
from art.utils import load_mnist
from art.visualization import create_sprite, convert_to_rgb, save_image, plot_3d


import logging
import unittest

import numpy as np
#import keras
import tensorflow as tf

from art.attacks.evasion.adversarial_patch.adversarial_patch import (
    AdversarialPatch,
    AdversarialPatchNumpy,
    AdversarialPatchPyTorch,
)
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin

from tests.utils import TestBase, master_seed
from tests.utils import get_image_classifier_tf, get_image_classifier_kr
from tests.utils import get_tabular_classifier_kr, get_image_classifier_pt
#from tests.attacks.utils import backend_test_classifier_type_check_fail

logger = logging.getLogger(__name__)
from PIL import Image


# Step 1: Load the MNIST dataset

(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

# Step 2: Create the model

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D

from deepface.deepface.basemodels import ArcFace
from deepface.deepface.commons import functions

import matplotlib.pyplot as plt

#----------------------------------------------
#build face recognition model



class FaceClassifier(Model):
    """
    Standard TensorFlow model for unit testing.
    """

    def __init__(self):
        super(FaceClassifier, self).__init__()
        #self.conv1 = Conv2D(filters=4, kernel_size=5, activation="relu")
        #self.conv2 = Conv2D(filters=10, kernel_size=5, activation="relu")
        #self.maxpool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid", data_format=None)
        #self.flatten = Flatten()
        #self.dense1 = Dense(100, activation="relu")
        #self.logits = Dense(10, activation="linear")


        self.embedding = ArcFace.loadModel()
        input_shape = self.embedding.layers[0].input_shape[0][1:3]


        img1 = functions.preprocess_face("dataset/img1.jpg", input_shape)
        #img1_representation = model.predict(img1)[0,:]
        #self.img1_feat = self.embedding.predict(img1)[0,:]
        self.img1_feat = self.embedding(img1)[0,:]

        #img2 = functions.preprocess_face("dataset/img3.jpg", input_shape)
        img2 = functions.preprocess_face("dataset/img19.jpg", input_shape)
        #self.img2_feat = self.embedding.predict(img2)[0,:]
        self.img2_feat = self.embedding(img2)[0,:]

        #img3 = functions.preprocess_face("dataset/img8.jpg", input_shape)
        img3 = functions.preprocess_face("dataset/img30.jpg", input_shape)
        #self.img3_feat = self.embedding.predict(img3)[0,:]
        self.img3_feat = self.embedding(img3)[0,:]


        #img4 = functions.preprocess_face("dataset/img22.jpg", input_shape)
        img4 = functions.preprocess_face("dataset/img61.jpg", input_shape)
        self.img4_feat = self.embedding(img4)[0,:]

        img5 = functions.preprocess_face("dataset/img20.jpg", input_shape)
        self.img5_feat = self.embedding(img5)[0,:]


    def get_logits(self, x):


        '''
        dist_vector1 = np.square(x - self.img1_feat)
        d1 = np.exp( -np.sqrt(dist_vector1.sum()))

        dist_vector2 = np.square(x - self.img2_feat)
        d2 = np.exp( -np.sqrt(dist_vector2.sum()))

        dist_vector3 = np.square(x - self.img3_feat)
        d3 = np.exp( -np.sqrt(dist_vector3.sum()))

        #d_all = np.array([d1, d2, d3])
        d_nor = d_all / np.sum(d_all)

        '''
        #print(x.shape)

        flag = 0
        for xi in x:

            d_v1 = tf.square(xi - self.img1_feat)
            d1 = tf.exp( -tf.sqrt(tf.reduce_sum(d_v1)))


            d_v2 = tf.square(xi - self.img2_feat)
            d2 = tf.exp( -tf.sqrt(tf.reduce_sum(d_v2)))

            d_v3 = tf.square(xi - self.img3_feat)
            d3 = tf.exp( -tf.sqrt(tf.reduce_sum(d_v3)))

            d_v4 = tf.square(xi - self.img4_feat)
            d4 = tf.exp( -tf.sqrt(tf.reduce_sum(d_v4)))

            d_v5 = tf.square(xi - self.img5_feat)
            d5 = tf.exp( -tf.sqrt(tf.reduce_sum(d_v5)))

            d_all = tf.stack([d1, d2, d3, d4, d5])

            d_nor = d_all / tf.reduce_sum(d_all)

            d_nor = tf.expand_dims(d_nor, 0)

            flag += 1
            #print(d_nor)
            if flag==1:
                d_nor_batch = d_nor
            else:
                #print(d_nor_batch.shape)
                d_nor_batch = tf.concat([d_nor_batch, d_nor], axis=0)

        #print(d_nor_batch.shape)


        #dis = [d1, d2, d3]
        #return torch.stack(dis)

        #return d_nor
        return d_nor_batch




    def call(self, x):
        #x = self.conv1(x)
        #x = self.maxpool(x)
        #x = self.flatten(x)
        #x = self.logits(x)

        #x = self.embedding.predict(x)[0, :]
        #x = self.embedding(x)[0]
        #print('fw')
        #print(x.shape)
        x = self.embedding(x)
        #print(x.shape)
        x = self.get_logits(x)
        #print(x.shape)
        #print(x)
        #x = softmax(x)
        return x

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))



Embedding = ArcFace.loadModel()
input_shape = Embedding.layers[0].input_shape[0][1:3]
print("model input shape: ", Embedding.layers[0].input_shape[1:])
print("model output shape: ", Embedding.layers[-1].input_shape[-1])

img1 = functions.preprocess_face("dataset/img1.jpg", input_shape)
#img2 = functions.preprocess_face("dataset/img3.jpg", input_shape)
img2 = functions.preprocess_face("dataset/img19.jpg", input_shape)
#img3 = functions.preprocess_face("dataset/img8.jpg", input_shape)
img3 = functions.preprocess_face("dataset/img30.jpg", input_shape)
#img4 = functions.preprocess_face("dataset/img22.jpg", input_shape)
img4 = functions.preprocess_face("dataset/img61.jpg", input_shape)
img5 = functions.preprocess_face("dataset/img20.jpg", input_shape)


test1_crop = functions.preprocess_face("dataset/img2.jpg", input_shape)
#test1_crop = functions.preprocess_face("dataset/img1.jpg", input_shape)


#model = TensorFlowModel()
#loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)


model = FaceClassifier()

#opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
#criterion = nn.NLLLoss()
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)



# Step 3: Create the ART classifier

classifier = TensorFlowV2Classifier(
    model=model,
    loss_object=loss_object,
    train_step=train_step,
    nb_classes=5,
    input_shape=(112, 112, 3),
    clip_values=(0, 1),
)


#----------------
##UAP

face1 = functions.preprocess_face("dataset/img1.jpg", input_shape)
face2 = functions.preprocess_face("dataset/img2.jpg", input_shape)
face3 = functions.preprocess_face("dataset/img4.jpg", input_shape)
face4 = functions.preprocess_face("dataset/img5.jpg", input_shape)
face5 = functions.preprocess_face("dataset/img6.jpg", input_shape)
face6 = functions.preprocess_face("dataset/img7.jpg", input_shape)
face7 = functions.preprocess_face("dataset/img10.jpg", input_shape)
face8 = functions.preprocess_face("dataset/img11.jpg", input_shape)


face_set = np.vstack([face1, face2, face3, face4, face5, face6, face7, face8])

#tlabel = [0.,  1., 0., 0., 0.]
tlabel = [0,  1,  0,  0,  0]
label_set = np.vstack([tlabel, tlabel, tlabel, tlabel, tlabel, tlabel, tlabel, tlabel])
print(tlabel)


#eps=0.1
#attack = UniversalPerturbation(classifier, attacker='pgd', eps=eps, max_iter=40)
#x_test_adv = attack.generate(x=face_set)

eps=0.9
attack = TargetedUniversalPerturbation(classifier, attacker='fgsm', eps=eps, max_iter=40)
#attack = TargetedUniversalPerturbation(classifier, attacker='simba', eps=eps, max_iter=40)
x_test_adv = attack.generate(x=face_set, y=label_set)


print('--')
print(x_test_adv.shape)

out_labels = []
diff = []
pertb = []
for i in range(8):
    prediction = classifier.predict(x_test_adv[i].reshape((1,112,112,3)))
    out_labels.append( np.argmax(prediction, axis=1)[0] )
    print(prediction)
    pertb.append( np.abs(face_set[i] - x_test_adv[i]) )
    diff.append( np.sum(np.abs(face_set[i] - x_test_adv[i])) )

print(out_labels)
print(diff)

fig = plt.figure()


ax1 = fig.add_subplot(2,7,1)
plt.imshow(img1[0][:,:,::-1])
plt.axis('off')
plt.text(0.1, 0.25, "target face ID: %s" % (tlabel ))

ax1 = fig.add_subplot(2,7,2)
plt.imshow(img2[0][:,:,::-1])
plt.axis('off')
ax1 = fig.add_subplot(2,7,3)
plt.imshow(img3[0][:,:,::-1])
plt.axis('off')
ax1 = fig.add_subplot(2,7,4)
plt.imshow(img4[0][:,:,::-1])
plt.axis('off')
ax1 = fig.add_subplot(2,7,5)
plt.imshow(img5[0][:,:,::-1])
plt.axis('off')


ax1 = fig.add_subplot(2,7,7)
plt.imshow(pertb[0][:,:,::-1])
plt.axis('off')
plt.text(0.1, 0.25, "With eps=%s" % (eps))

ax1 = fig.add_subplot(2,8,8+1)
plt.imshow(x_test_adv[0][:,:,::-1])
#plt.imshow(pertb[0][:,:,::-1])
plt.axis('off')
plt.text(0.1, 0.25, "output labels: %s" % (out_labels))
ax1 = fig.add_subplot(2,8,8+2)
plt.imshow(x_test_adv[1][:,:,::-1])
#plt.imshow(pertb[1][:,:,::-1])
plt.axis('off')
ax1 = fig.add_subplot(2,8,8+3)
plt.imshow(x_test_adv[2][:,:,::-1])
#plt.imshow(pertb[2][:,:,::-1])
plt.axis('off')
ax1 = fig.add_subplot(2,8,8+4)
plt.imshow(x_test_adv[3][:,:,::-1])
plt.axis('off')
ax1 = fig.add_subplot(2,8,8+5)
plt.imshow(x_test_adv[4][:,:,::-1])
plt.axis('off')
ax1 = fig.add_subplot(2,8,8+6)
plt.imshow(x_test_adv[5][:,:,::-1])
plt.axis('off')
ax1 = fig.add_subplot(2,8,8+7)
plt.imshow(x_test_adv[6][:,:,::-1])
plt.axis('off')
ax1 = fig.add_subplot(2,8,8+8)
plt.imshow(x_test_adv[7][:,:,::-1])
plt.axis('off')
plt.show()
print(over)


#----------------------






patch_size = 40

attack_ap = AdversarialPatch(
            classifier,
            rotation_max=0.5,
            scale_min=0.4,
            scale_max=0.41,
            learning_rate=5.0,
            #learning_rate=0.1,
            batch_size=10,
            max_iter=40,
            #patch_shape=(28, 28, 1),
            patch_shape=(patch_size, patch_size, 3),
            verbose=False,
            )


face1 = functions.preprocess_face("dataset/img1.jpg", input_shape)
face2 = functions.preprocess_face("dataset/img2.jpg", input_shape)
face3 = functions.preprocess_face("dataset/img4.jpg", input_shape)
face4 = functions.preprocess_face("dataset/img5.jpg", input_shape)
face5 = functions.preprocess_face("dataset/img6.jpg", input_shape)
face6 = functions.preprocess_face("dataset/img7.jpg", input_shape)
face7 = functions.preprocess_face("dataset/img10.jpg", input_shape)
face8 = functions.preprocess_face("dataset/img11.jpg", input_shape)


face_set = np.vstack([face1, face2, face3, face4, face5, face6, face7, face8])

tlabel = [0,  0, 0, 0, 1]
label_set = np.vstack([tlabel, tlabel, tlabel, tlabel, tlabel, tlabel, tlabel, tlabel])




print(face_set.shape)
print(label_set.shape)


#mask = np.ones((8, 112, 112)).astype(bool)
mask = np.zeros((8, 112, 112)).astype(bool)
mask[:, 0:30, 20:90] = np.ones((8, 30, 70)).astype(bool)

patch_adv, patch_mask = attack_ap.generate(x=face_set, y=label_set, mask=mask, shuffle=False)
#patch_adv, patch_mask = attack_ap.generate(face_set, label_set, shuffle=False)
#patch_adv, patch_mask = attack_ap.generate(face_set, shuffle=False)

print('---')
print(patch_adv.shape)
print(patch_mask.shape)
#x_out = attack_ap.apply_patch(x=face_set, scale=0.4, mask=mask)
#print(x_out.shape)



attack_ap.reset_patch(initial_patch_value=patch_adv)


prediction, patched_input = attack_ap._attack._predictions_input(images=face_set, mask=mask)
print(prediction)
rst = np.argmax(prediction, axis=1)
print(rst)

print(patched_input.shape)



#----------------------------------------------
#plotting
print('===')

'''
ii = (face_set[0]*255).astype(np.uint8)
print(ii.shape)
print(ii)
iii = ii[:,:,::-1]
print(iii.shape)
print(iii)
im = Image.fromarray(iii)
im.show()
'''

if 1:
        fig = plt.figure()

        ax1 = fig.add_subplot(3,8,1)
        plt.imshow(patch_adv[:,:,::-1])
        #plt.imshow(patch_adv)
        plt.axis('off')
        plt.text(0.1, 0.25, "patch size is: %s " % (patch_size))

        ax1 = fig.add_subplot(3,8,2)
        plt.imshow(mask[0])
        #plt.imshow(patch_mask[:,:,::-1])
        plt.axis('off')


        ax1 = fig.add_subplot(3,8,4)
        plt.imshow(img1[0][:,:,::-1])
        plt.axis('off')
        plt.text(0.1, 0.25, "target: %s from 5 face IDs" % (tlabel))
        ax1 = fig.add_subplot(3,8,5)
        plt.imshow(img2[0][:,:,::-1])
        plt.axis('off')
        ax1 = fig.add_subplot(3,8,6)
        plt.imshow(img3[0][:,:,::-1])
        plt.axis('off')
        ax1 = fig.add_subplot(3,8,7)
        plt.imshow(img4[0][:,:,::-1])
        plt.axis('off')
        ax1 = fig.add_subplot(3,8,8)
        plt.imshow(img5[0][:,:,::-1])
        plt.axis('off')



        ax1 = fig.add_subplot(3,8,8+1)
        plt.imshow(face_set[0][:,:,::-1])
        plt.axis('off')
        plt.text(0.1, 0.25, "input face set")
        ax1 = fig.add_subplot(3,8,8+2)
        plt.imshow(face_set[1][:,:,::-1])
        plt.axis('off')
        ax1 = fig.add_subplot(3,8,8+3)
        plt.imshow(face_set[2][:,:,::-1])
        plt.axis('off')

        ax1 = fig.add_subplot(3,8,8+4)
        plt.imshow(face_set[3][:,:,::-1])
        plt.axis('off')
        ax1 = fig.add_subplot(3,8,8+5)
        plt.imshow(face_set[4][:,:,::-1])
        plt.axis('off')
        ax1 = fig.add_subplot(3,8,8+6)
        plt.imshow(face_set[5][:,:,::-1])
        plt.axis('off')
        ax1 = fig.add_subplot(3,8,8+7)
        plt.imshow(face_set[6][:,:,::-1])
        plt.axis('off')
        ax1 = fig.add_subplot(3,8,8+8)
        plt.imshow(face_set[7][:,:,::-1])
        plt.axis('off')













        ax1 = fig.add_subplot(3,8,2*8+1)
        plt.imshow(patched_input[0][:,:,::-1])
        #plt.imshow(x_out[0][:,:,::-1])
        #plt.imshow(patch_mask)
        plt.axis('off')
        plt.text(0.1, 0.25, "succeed: %s" % (rst))

        ax1 = fig.add_subplot(3,8,2*8+2)
        plt.imshow(patched_input[1][:,:,::-1])
        plt.axis('off')

        ax1 = fig.add_subplot(3,8,2*8+3)
        plt.imshow(patched_input[2][:,:,::-1])
        plt.axis('off')

        ax1 = fig.add_subplot(3,8,2*8+4)
        plt.imshow(patched_input[3][:,:,::-1])
        plt.axis('off')
        ax1 = fig.add_subplot(3,8,2*8+5)
        plt.imshow(patched_input[4][:,:,::-1])
        plt.axis('off')
        ax1 = fig.add_subplot(3,8,2*8+6)
        plt.imshow(patched_input[5][:,:,::-1])
        plt.axis('off')
        ax1 = fig.add_subplot(3,8,2*8+7)
        plt.imshow(patched_input[6][:,:,::-1])
        plt.axis('off')
        ax1 = fig.add_subplot(3,8,2*8+8)
        plt.imshow(patched_input[7][:,:,::-1])
        plt.axis('off')



        plt.show()


