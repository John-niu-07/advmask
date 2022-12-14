# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import numpy as np
import keras
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
from tests.attacks.utils import backend_test_classifier_type_check_fail

logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
from PIL import Image

class TestAdversarialPatch(TestBase):
    """
    A unittest class for testing Adversarial Patch attack.
    """

    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234)
        super().setUpClass()

        cls.n_train = 1000
        cls.n_test = 1
        cls.x_train_mnist = cls.x_train_mnist[0 : cls.n_train]
        cls.y_train_mnist = cls.y_train_mnist[0 : cls.n_train]
        #cls.x_train_mnist = cls.x_train_mnist[2 : 2+cls.n_train]
        #cls.y_train_mnist = cls.y_train_mnist[2 : 2+cls.n_train]
        cls.x_test_mnist = cls.x_test_mnist[0 : cls.n_test]
        cls.y_test_mnist = cls.y_test_mnist[0 : cls.n_test]

    def setUp(self):
        master_seed(seed=1234)
        super().setUp()

    def test_2_tensorflow_numpy(self):
        """
        First test with the TensorFlowClassifier.
        :return:
        """
        import tensorflow as tf

        tfc, sess = get_image_classifier_tf(from_logits=True)

        attack_ap = AdversarialPatchNumpy(
            tfc,
            rotation_max=0.5,
            scale_min=0.4,
            scale_max=0.41,
            learning_rate=5.0,
            batch_size=10,
            max_iter=5,
            verbose=False,
        )

        target = np.zeros(self.x_train_mnist.shape[0])
        patch_adv, _ = attack_ap.generate(self.x_train_mnist, target, shuffle=False)

        if tf.__version__[0] == "2":
            self.assertAlmostEqual(patch_adv[8, 8, 0], 0.67151666, delta=0.05)
            self.assertAlmostEqual(patch_adv[14, 14, 0], 0.6292826, delta=0.05)
            self.assertAlmostEqual(float(np.sum(patch_adv)), 424.31439208984375, delta=1.0)
        else:
            self.assertAlmostEqual(patch_adv[8, 8, 0], 0.67151666, delta=0.05)
            self.assertAlmostEqual(patch_adv[14, 14, 0], 0.6292826, delta=0.05)
            self.assertAlmostEqual(float(np.sum(patch_adv)), 424.31439208984375, delta=1.0)

        # insert_transformed_patch
        x_out = attack_ap.insert_transformed_patch(
            self.x_train_mnist[0], np.ones((14, 14, 1)), np.asarray([[2, 13], [2, 18], [12, 22], [8, 13]])
        )
        x_out_expexted = np.array(
            [
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.84313726,
                0.0,
                0.0,
                0.0,
                0.0,
                0.1764706,
                0.7294118,
                0.99215686,
                0.99215686,
                0.5882353,
                0.10588235,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        np.testing.assert_almost_equal(x_out[15, :, 0], x_out_expexted, decimal=3)

        if sess is not None:
            sess.close()

    @unittest.skipIf(int(tf.__version__.split(".")[0]) != 2, reason="Skip unittests if not TensorFlow>=2.0.")
    def test_3_tensorflow_v2_framework(self):
        """
        First test with the TensorFlowClassifier.
        :return:
        """
        tfc, _ = get_image_classifier_tf(from_logits=True)

        #x_train = np.reshape(self.x_train_mnist, (self.n_train, 1, 28, 28)).astype(np.float32)

        tfc.fit(self.x_train_mnist, self.y_train_mnist, batch_size=256, nb_epochs=100)

        predictions = tfc.predict(self.x_train_mnist)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(self.y_train_mnist, axis=1)) / len(self.y_train_mnist)
        print("FT accuracy on benign test examples: {}%".format(accuracy * 100))

        predictions = tfc.predict(self.x_test_mnist)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(self.y_test_mnist, axis=1)) / len(self.y_test_mnist)
        print("FT accuracy on benign test examples: {}%".format(accuracy * 100))
        print(predictions)
        print(np.argmax(predictions, axis=1))



        attack_ap = AdversarialPatch(
            tfc,
            rotation_max=0.5,
            scale_min=0.8,
            scale_max=0.81,
            learning_rate=5.0,
            #learning_rate=0.1,
            batch_size=10,
            max_iter=10,
            #patch_shape=(28, 28, 1),
            patch_shape=(18, 18, 1),
            verbose=False,
        )


        

        #targeted attack
        #target = np.zeros(self.x_train_mnist.shape[0])
        target = np.zeros(self.y_train_mnist.shape)
        print(target.shape)
        target[:,3] = np.ones((self.y_train_mnist.shape[0]))
        print(target)
        print(self.x_train_mnist.shape)
        patch_adv, patch_mask = attack_ap.generate(self.x_train_mnist, target, shuffle=False)

        #untargeted attack
        #patch_adv, patch_mask = attack_ap.generate(self.x_train_mnist, shuffle=False)


        print(patch_adv.shape)
        print(patch_mask.shape)
        #print(patch_mask)

        #self.assertAlmostEqual(patch_adv[8, 8, 0], 1.0, delta=0.05)
        #self.assertAlmostEqual(patch_adv[14, 14, 0], 0.0, delta=0.05)
        #self.assertAlmostEqual(float(np.sum(patch_adv)), 377.415771484375, delta=1.0)


        '''
        attack_ap.reset_patch(initial_patch_value=patch_adv)
        mask = np.ones((1, 28, 28)).astype(bool)
        x_out = attack_ap.apply_patch(x=self.x_train_mnist, scale=0.01, mask=mask)
        prediction = attack_ap._attack._predictions(images=x_out, mask=mask)
        print(prediction)
        '''

        attack_ap.reset_patch(initial_patch_value=patch_adv)

        mask = np.ones((1, 28, 28)).astype(bool)
        x_out = attack_ap.apply_patch(x=self.x_test_mnist, scale=0.8, mask=mask)


        #ii = (self.x_train_mnist[0]*255).reshape((28,28)).astype(np.uint8)
        #im = Image.fromarray(ii)
        #im.show()


        prediction = attack_ap._attack._predictions(images=x_out, mask=mask)
        print(prediction)
        print(np.argmax(prediction, axis=1))

        fig = plt.figure()
        
        ax1 = fig.add_subplot(1,4,1)
        plt.imshow(patch_adv)
        plt.axis('off')

        ax1 = fig.add_subplot(1,4,2)
        plt.imshow(patch_mask)
        plt.axis('off')

        ax1 = fig.add_subplot(1,4,3)
        plt.imshow(self.x_test_mnist[0])
        plt.axis('off')

        ax1 = fig.add_subplot(1,4,4)
        plt.imshow(x_out[0])
        #plt.imshow(x_out[0], cmap='gray')
        #plt.imshow(patch_mask)
        plt.axis('off')
        plt.show()
        

        mask = np.ones((1, 28, 28)).astype(bool)
        attack_ap.apply_patch(x=self.x_train_mnist, scale=0.1, mask=mask)
        attack_ap.reset_patch(initial_patch_value=None)
        attack_ap.reset_patch(initial_patch_value=1.0)
        attack_ap.reset_patch(initial_patch_value=patch_adv)

    @unittest.skipIf(
        int(keras.__version__.split(".")[0]) == 2 and int(keras.__version__.split(".")[1]) < 3,
        reason="Skip unittests if not Keras>=2.3.",
    )
    def test_6_keras(self):
        """
        Second test with the KerasClassifier.
        :return:
        """
        krc = get_image_classifier_kr(from_logits=True)

        attack_ap = AdversarialPatch(
            krc,
            rotation_max=0.5,
            scale_min=0.4,
            scale_max=0.41,
            learning_rate=5.0,
            batch_size=10,
            max_iter=5,
            verbose=False,
        )

        target = np.zeros(self.x_train_mnist.shape[0])
        patch_adv, _ = attack_ap.generate(self.x_train_mnist, target)

        self.assertAlmostEqual(patch_adv[8, 8, 0], 0.67151666, delta=0.05)
        self.assertAlmostEqual(patch_adv[14, 14, 0], 0.6292826, delta=0.05)
        self.assertAlmostEqual(float(np.sum(patch_adv)), 424.31439208984375, delta=1.0)

        # insert_transformed_patch
        x_out = attack_ap.insert_transformed_patch(
            self.x_train_mnist[0], np.ones((14, 14, 1)), np.asarray([[2, 13], [2, 18], [12, 22], [8, 13]])
        )
        x_out_expexted = np.array(
            [
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.84313726,
                0.0,
                0.0,
                0.0,
                0.0,
                0.1764706,
                0.7294118,
                0.99215686,
                0.99215686,
                0.5882353,
                0.10588235,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        np.testing.assert_almost_equal(x_out[15, :, 0], x_out_expexted, decimal=3)

    def test_4_pytorch(self):
        """
        Third test with the PyTorchClassifier.
        :return:
        """
        ptc = get_image_classifier_pt(from_logits=True)

        x_train = np.reshape(self.x_train_mnist, (self.n_train, 1, 28, 28)).astype(np.float32)

        attack_ap = AdversarialPatch(
            ptc,
            rotation_max=0.5,
            scale_min=0.4,
            scale_max=0.41,
            learning_rate=5.0,
            batch_size=10,
            max_iter=5,
            patch_shape=(1, 28, 28),
            verbose=False,
        )

        target = np.zeros(self.x_train_mnist.shape[0])
        #patch_adv, patch_mask = attack_ap.generate(x_train, target)
        patch_adv, patch_mask = attack_ap.generate(x_train)

        print(x_train.shape)

        print(patch_adv.shape)
        print(patch_mask.shape)

        #self.assertAlmostEqual(patch_adv[0, 8, 8], 0.5, delta=0.05)
        #self.assertAlmostEqual(patch_adv[0, 14, 14], 0.5, delta=0.05)
        #self.assertAlmostEqual(float(np.sum(patch_adv)), 371.88014772999827, delta=4.0)


        attack_ap.reset_patch(initial_patch_value=patch_adv)

        mask = np.ones((1, 28, 28)).astype(bool)
        x_out = attack_ap.apply_patch(x=x_train, scale=0.4, mask=mask)

        prediction = attack_ap._attack._predictions(images=x_out, mask=mask)
        print(prediction)

        print(x_out.shape)
        fig = plt.figure()

        ax1 = fig.add_subplot(1,4,1)
        plt.imshow(patch_adv)
        plt.axis('off')

        ax1 = fig.add_subplot(1,4,2)
        plt.imshow(patch_mask)
        plt.axis('off')

        ax1 = fig.add_subplot(1,4,3)
        plt.imshow(self.x_train_mnist[0])
        plt.axis('off')

        ax1 = fig.add_subplot(1,4,4)
        plt.imshow(x_out[0])
        #plt.imshow(patch_mask)
        plt.axis('off')
        plt.show()




        mask = np.ones((1, 28, 28)).astype(bool)
        attack_ap.apply_patch(x=x_train, scale=0.1, mask=mask)
        attack_ap.reset_patch(initial_patch_value=None)
        attack_ap.reset_patch(initial_patch_value=1.0)
        attack_ap.reset_patch(initial_patch_value=patch_adv)
        with self.assertRaises(ValueError):
            attack_ap.reset_patch(initial_patch_value=np.array([1, 2, 3]))

        # Numpy
        attack_ap = AdversarialPatchNumpy(
            ptc,
            rotation_max=0.5,
            scale_min=0.4,
            scale_max=0.41,
            learning_rate=5.0,
            batch_size=10,
            max_iter=5,
            verbose=False,
        )

        target = np.zeros(self.x_train_mnist.shape[0])
        patch_adv, _ = attack_ap.generate(x_train, target)

        self.assertAlmostEqual(patch_adv[0, 8, 8], 0.6715167, delta=0.05)
        self.assertAlmostEqual(patch_adv[0, 14, 14], 0.6292826, delta=0.05)
        self.assertAlmostEqual(float(np.sum(patch_adv)), 424.31439208984375, delta=4.0)

        mask = np.ones((1, 28, 28)).astype(bool)
        attack_ap.apply_patch(x=x_train, scale=0.1, mask=mask)
        attack_ap.reset_patch(initial_patch_value=None)
        attack_ap.reset_patch(initial_patch_value=1.0)
        attack_ap.reset_patch(initial_patch_value=patch_adv)
        with self.assertRaises(ValueError):
            attack_ap.reset_patch(initial_patch_value=np.array([1, 2, 3]))

    def test_5_failure_feature_vectors(self):
        classifier = get_tabular_classifier_kr()
        classifier._clip_values = (0, 1)

        # Assert that value error is raised for feature vectors
        with self.assertRaises(ValueError) as context:
            _ = AdversarialPatch(classifier=classifier)

        self.assertIn(
            "Unexpected input_shape in estimator detected. AdversarialPatch is expecting images or videos as input.",
            str(context.exception),
        )

    def test_check_params(self):

        ptc = get_image_classifier_pt(from_logits=True)

        krc = get_image_classifier_kr(from_logits=True)

        # AdversarialPatch

        with self.assertRaises(ValueError):
            _ = AdversarialPatch(ptc, rotation_max="1")
        with self.assertRaises(ValueError):
            _ = AdversarialPatch(ptc, rotation_max=-1)

        with self.assertRaises(ValueError):
            _ = AdversarialPatch(ptc, scale_min="1")
        with self.assertRaises(ValueError):
            _ = AdversarialPatch(ptc, scale_min=-1.0)

        with self.assertRaises(ValueError):
            _ = AdversarialPatch(ptc, scale_max=1)
        with self.assertRaises(ValueError):
            _ = AdversarialPatch(ptc, scale_max=2.0)

        with self.assertRaises(ValueError):
            _ = AdversarialPatch(ptc, learning_rate=1)
        with self.assertRaises(ValueError):
            _ = AdversarialPatch(krc, learning_rate=-1.0)

        with self.assertRaises(ValueError):
            _ = AdversarialPatch(ptc, max_iter=1.0)
        with self.assertRaises(ValueError):
            _ = AdversarialPatch(ptc, max_iter=-1)

        with self.assertRaises(ValueError):
            _ = AdversarialPatch(ptc, batch_size=1.0)
        with self.assertRaises(ValueError):
            _ = AdversarialPatch(ptc, batch_size=-1)

        with self.assertRaises(ValueError):
            _ = AdversarialPatch(ptc, verbose="true")

        # AdversarialPatchPyTorch

        with self.assertRaises(ValueError):
            _ = AdversarialPatchPyTorch(ptc, distortion_scale_max="1")

        with self.assertRaises(ValueError):
            _ = AdversarialPatchPyTorch(ptc, patch_type="triangle")

        # AdversarialPatchNumpy

        with self.assertRaises(ValueError):
            _ = AdversarialPatchNumpy(ptc, rotation_max="1")
        with self.assertRaises(ValueError):
            _ = AdversarialPatchNumpy(ptc, rotation_max=-1)

        with self.assertRaises(ValueError):
            _ = AdversarialPatchNumpy(ptc, scale_min="1")
        with self.assertRaises(ValueError):
            _ = AdversarialPatchNumpy(ptc, scale_min=-1.0)

        with self.assertRaises(ValueError):
            _ = AdversarialPatchNumpy(ptc, scale_max=1)
        with self.assertRaises(ValueError):
            _ = AdversarialPatchNumpy(ptc, scale_max=2.0)

        with self.assertRaises(ValueError):
            _ = AdversarialPatchNumpy(ptc, learning_rate="1")
        with self.assertRaises(ValueError):
            _ = AdversarialPatchNumpy(krc, learning_rate=-1.0)

        with self.assertRaises(ValueError):
            _ = AdversarialPatchNumpy(ptc, max_iter=1.0)
        with self.assertRaises(ValueError):
            _ = AdversarialPatchNumpy(ptc, max_iter=-1)

        with self.assertRaises(ValueError):
            _ = AdversarialPatchNumpy(ptc, batch_size=1.0)
        with self.assertRaises(ValueError):
            _ = AdversarialPatchNumpy(ptc, batch_size=-1)

        with self.assertRaises(ValueError):
            _ = AdversarialPatchNumpy(ptc, verbose="true")

    def test_1_classifier_type_check_fail(self):
        backend_test_classifier_type_check_fail(AdversarialPatch, [BaseEstimator, NeuralNetworkMixin, ClassifierMixin])


        print('adv_patch_tf_generate')

tt = TestAdversarialPatch()

tt.setUp()
tt.setUpClass()
#tt.test_4_pytorch()
tt.test_3_tensorflow_v2_framework()


#if __name__ == "__main__":
#    unittest.main()
