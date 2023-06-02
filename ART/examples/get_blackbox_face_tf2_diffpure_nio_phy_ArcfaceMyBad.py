"""
The script demonstrates a simple example of using ART with TensorFlow v1.x. The example train a small model on the MNIST
dataset and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train
the model, it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import numpy as np

from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
from art.attacks.evasion.hop_skip_jump import HopSkipJump
from art.attacks.evasion.geometric_decision_based_attack import GeoDA
from art.attacks.evasion.universal_perturbation import UniversalPerturbation
from art.attacks.evasion.targeted_universal_perturbation import TargetedUniversalPerturbation


from art.estimators.classification import TensorFlowV2Classifier
from art.utils import load_mnist
from art.visualization import create_sprite, convert_to_rgb, save_image, plot_3d



# Step 1: Load the MNIST dataset

#(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

# Step 2: Create the model

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D

from deepface.deepface.basemodels import ArcFace_my
from deepface.deepface.commons import functions

import matplotlib.pyplot as plt

from skimage.transform import resize
from torchvision import transforms
import cv2



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


        self.embedding = ArcFace_my.loadModel()
        input_shape = (112, 112)


        img1 = functions.preprocess_face_woDet("/face/Mask/idinvert_pytorch/examples/r256/008_aligned.png", input_shape)
        #img1_representation = model.predict(img1)[0,:]
        #self.img1_feat = self.embedding.predict(img1)[0,:]
        self.img1_feat = self.embedding(transforms.ToTensor()(img1[0]).to('cuda').unsqueeze(0))[0,:] 

        img2 = functions.preprocess_face_woDet("/face/Mask/idinvert_pytorch/examples/r256/009_aligned.png", input_shape)
        img2 = functions.preprocess_face_woDet("/face/Mask/AdversarialMask/datasets/CASIA/4204960/085_aligned.png", input_shape)
        #self.img2_feat = self.embedding.predict(img2)[0,:]
        self.img2_feat = self.embedding(transforms.ToTensor()(img2[0]).to('cuda').unsqueeze(0))[0,:] 

        img3 = functions.preprocess_face_woDet("/face/Mask/idinvert_pytorch/examples/r256/010_aligned.png", input_shape)
        #img3 = functions.preprocess_face_woDet("/face/Mask/idinvert_pytorch/examples/r256/004_aligned.png", input_shape)
        #self.img3_feat = self.embedding.predict(img3)[0,:]
        self.img3_feat = self.embedding(transforms.ToTensor()(img3[0]).to('cuda').unsqueeze(0))[0,:]


        img4 = functions.preprocess_face_woDet("/face/Mask/idinvert_pytorch/examples/r256/003_aligned.png", input_shape)
        self.img4_feat = self.embedding(transforms.ToTensor()(img4[0]).to('cuda').unsqueeze(0))[0,:]

        #img5 = functions.preprocess_face("dataset/img20.jpg", input_shape)
        img5 = functions.preprocess_face_woDet("/face/Mask/idinvert_pytorch/examples/r256/007_aligned.png", input_shape)
        self.img5_feat = self.embedding(transforms.ToTensor()(img5[0]).to('cuda').unsqueeze(0))[0,:]


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

        d_v1 = tf.square(x - self.img1_feat)
        d1 = tf.exp( -tf.sqrt(tf.reduce_sum(d_v1)))


        d_v2 = tf.square(x - self.img2_feat)
        d2 = tf.exp( -tf.sqrt(tf.reduce_sum(d_v2)))

        d_v3 = tf.square(x - self.img3_feat)
        d3 = tf.exp( -tf.sqrt(tf.reduce_sum(d_v3)))

        d_v4 = tf.square(x - self.img4_feat)
        d4 = tf.exp( -tf.sqrt(tf.reduce_sum(d_v4)))

        d_v5 = tf.square(x - self.img5_feat)
        d5 = tf.exp( -tf.sqrt(tf.reduce_sum(d_v5)))

        d_all = tf.stack([d1, d2, d3, d4, d5])

        d_nor = d_all / tf.reduce_sum(d_all)

        d_nor = tf.expand_dims(d_nor, 0)

        #print(d_nor)
        
        #dis = [d1, d2, d3]
        #return torch.stack(dis)
        return d_nor
        #return d_all




    def call(self, x):
        #x = self.conv1(x)
        #x = self.maxpool(x)
        #x = self.flatten(x)
        #x = self.logits(x)

        #x = self.embedding.predict(x)[0, :]

        #print('===')
        #print(x.shape)
        ##x = tf.convert_to_tensor( resize(x, (1,112,112,3)), tf.float32 )
        x = tf.image.resize(x, [112,112])
        #print(x.shape)

        x = self.embedding(transforms.ToTensor()(x))[0]
        x = self.get_logits(x)
        #x = softmax(x)
        return x

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))



Embedding = ArcFace_my.loadModel()
input_shape = (112, 112)

print("model input shape: ", input_shape)
img1 = functions.preprocess_face_woDet("/face/Mask/idinvert_pytorch/examples/r256/008_aligned.png", input_shape)
img2 = functions.preprocess_face_woDet("/face/Mask/idinvert_pytorch/examples/r256/009_aligned.png", input_shape)
img2 = functions.preprocess_face_woDet("/face/Mask/AdversarialMask/datasets/CASIA/4204960/085_aligned.png", input_shape)

img3 = functions.preprocess_face_woDet("/face/Mask/idinvert_pytorch/examples/r256/010_aligned.png", input_shape)
#img3 = functions.preprocess_face_woDet("/face/Mask/idinvert_pytorch/examples/r256/004_aligned.png", input_shape)

img4 = functions.preprocess_face_woDet("/face/Mask/idinvert_pytorch/examples/r256/003_aligned.png", input_shape)
img5 = functions.preprocess_face_woDet("/face/Mask/idinvert_pytorch/examples/r256/007_aligned.png", input_shape)


#test1_crop = functions.preprocess_face("dataset/img2.jpg", input_shape)
#test1_crop = functions.preprocess_face("dataset/img1.jpg", input_shape)

input_shape =(256, 256)
#input_shape =(112, 112)
print(input_shape)
##test1_crop = functions.preprocess_face_woDet("/face/Mask/AdversarialMask/datasets/CASIA/1302735/058_aligned.png", input_shape)
#test1_crop = functions.preprocess_face_woDet("/face/Mask/idinvert_pytorch/examples/r256/examples/bb_9_p_aligned.png", input_shape)
#test1_crop = functions.preprocess_face_woDet("/face/Mask/idinvert_pytorch/examples/r256/examples/bb_9_s_aligned.png", input_shape)

#test1_crop = functions.preprocess_face_woDet("/face/Mask/DiffusionCLIP/runs/test_FT_adv_CelebA_HQ_beards_id_0.01_l1_0.0_clip_0/image_samples/train_1_person_with_beards_14_ngen6.png", input_shape)
#test1_crop = functions.preprocess_face_woDet("/face/Mask/DiffusionCLIP/runs/test_FT_adv_CelebA_HQ_beards_id_0.1_l1_0.0_clip_0/image_samples/train_1_person_with_beards_14_ngen6.png", input_shape)
test1_crop = functions.preprocess_face_woDet("/face/Mask/DiffusionCLIP/runs/test_FT_adv_CelebA_HQ_beards_id_1.0_l1_0.0_clip_0/image_samples/train_1_person_with_beards_10_ngen6.png", input_shape)

img_save = (test1_crop[0][:,:,::-1]*255).astype(np.uint8)
#save_image(img_save, "/sdiff/DiffPure/advimg/img.png")
save_image(img_save, "/sdiff/DiffPure/advimg/advimg.png")






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
    #input_shape=(112, 112, 3),
    input_shape=(256, 256, 3),
    clip_values=(0, 1),
)

# Step 4: Train the ART classifier

'''
classifier.fit(x_train, y_train, batch_size=64, nb_epochs=3)

# Step 5: Evaluate the ART classifier on benign test examples

predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))
'''

'''
attack = HopSkipJump(
            classifier,
            norm=np.inf,
            #norm=2,
            max_iter=10,
            #max_iter=10,
            #targeted=False,
            targeted=True, #Cannot
            #max_eval=5000,
            max_eval=500,
            init_eval=100,
            init_size=100,
            batch_size=1,
            verbose=True,
            )



attack = GeoDA(
            classifier,
            norm=np.inf,
            max_iter=4000,
            batch_size=1,
            verbose=False,
            )
'''


#attack = FastGradientMethod(estimator=classifier, eps=0.2)

attack = ProjectedGradientDescent(
            classifier,
            norm=np.inf,
            #eps=2.0 / 255.0,
            eps=8.0 / 255.0,
            eps_step=2.0 / 255.0,
            max_iter=40,
            #targeted=False,
            targeted=True,
            num_random_init=5,
            batch_size=1,
            )




print('')
print('')
print('==== 1. pred =======')
predictions = classifier.predict(test1_crop)
print(predictions)



print('')
print('=== /sdiff/DiffPure# python inference_run.py ===')
print('==== 3. before diffpure =======')
#test1_diffpure = cv2.imread("/sdiff/DiffPure/outputs/before_defended.png")
test1_diffpure_before = functions.preprocess_face_woDet("/sdiff/DiffPure/outputs/before_defended.png", input_shape)
diffpure_predictions = classifier.predict(test1_diffpure_before)
print(diffpure_predictions)


print('==== 3. after diffpure =======')
#test1_diffpure = cv2.imread("/sdiff/DiffPure/outputs/defended.png")
test1_diffpure_after = functions.preprocess_face_woDet("/sdiff/DiffPure/outputs/defended.png", input_shape)
diffpure_predictions = classifier.predict(test1_diffpure_after)
print(diffpure_predictions)
print('')
print('')










#----------------------------------------------
#plotting

fig = plt.figure()


ax1 = fig.add_subplot(2,5,1)
plt.imshow(img1[0][:,:,::-1])
plt.axis('off')

ax1 = fig.add_subplot(2,5,2)
plt.imshow(img2[0][:,:,::-1])
plt.axis('off')
ax1 = fig.add_subplot(2,5,3)
plt.imshow(img3[0][:,:,::-1])
plt.axis('off')
ax1 = fig.add_subplot(2,5,4)
plt.imshow(img4[0][:,:,::-1])
plt.axis('off')
ax1 = fig.add_subplot(2,5,5)
plt.imshow(img5[0][:,:,::-1])
plt.axis('off')

ax1 = fig.add_subplot(2,5,6)
plt.imshow(test1_crop[0][:,:,::-1])
plt.axis('off')



ax2 = fig.add_subplot(2,5,9)
plt.imshow(test1_diffpure_before[0][:,:,::-1])
plt.axis('off')

ax2 = fig.add_subplot(2,5,10)
plt.imshow(test1_diffpure_after[0][:,:,::-1])
plt.axis('off')

plt.show()

'''
# Step 7: Evaluate the ART classifier on adversarial test examples

predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
'''
