#python3 face_preparation.py '/art/adversarial-robustness-toolbox/examples/dataset/mat_hat.jpg'
#python3 face_preparation.py '/art/adversarial-robustness-toolbox/examples/dataset/j1.jpg'
#python3 face_preparation.py '/art/adversarial-robustness-toolbox/examples/dataset/j2.jpg'
#python3 face_preparation.py '/art/adversarial-robustness-toolbox/examples/dataset/jj2.jpg'
#python3 face_preparation.py '/art/adversarial-robustness-toolbox/examples/dataset/jj3.jpg'


#python3 face_preparation.py '/art/adversarial-robustness-toolbox/examples/dataset/mat_hat.jpg' --mask
#python3 face_preparation_my.py '/art/adversarial-robustness-toolbox/examples/dataset/john_hat.jpg' --mask
#python3 face_preparation_my2.py '/art/adversarial-robustness-toolbox/examples/dataset/john_hat.jpg' --mask


#python3 attack.py '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned.png'  '/face/hat/advhat/r100.pb' --anchor_face '/art/adversarial-robustness-toolbox/examples/dataset/j1_aligned.png' --init_face '/art/adversarial-robustness-toolbox/examples/dataset/j1_aligned.png'
#python3 attack.py '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned.png'  '/face/hat/advhat/r100.pb' --anchor_face '/art/adversarial-robustness-toolbox/examples/dataset/j1_aligned.png' --init_log '/face/hat/advhat/Attack/john_hatcrop.png'
#python3 attack.py '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned.png'  '/face/hat/advhat/r100.pb' --anchor_face '/art/adversarial-robustness-toolbox/examples/dataset/img2_aligned.png'
#python3 attack.py '/art/adversarial-robustness-toolbox/examples/dataset/mat_hat_aligned.png'  '/face/hat/advhat/r100.pb' 


#python3 face_preparation_add.py '/art/adversarial-robustness-toolbox/examples/dataset/mat_hat_aligned.png' --mask

#python3 attack_add.py '/art/adversarial-robustness-toolbox/examples/dataset/img32_aligned.png' '/face/hat/advhat/r100.pb' --hat_face '/art/adversarial-robustness-toolbox/examples/dataset/img18_mask_mask.png'  --target_face '/art/adversarial-robustness-toolbox/examples/dataset/img32_aligned.png' --orig_face '/art/adversarial-robustness-toolbox/examples/dataset/img18_aligned.png'



#python3 cos_tf.py '/art/adversarial-robustness-toolbox/examples/dataset/img18_aligned.png' '/art/adversarial-robustness-toolbox/examples/dataset/mat_hat_aligned.png' '/face/hat/advhat/r100.pb'
#python3 cos_tf.py '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned.png' '/art/adversarial-robustness-toolbox/examples/dataset/j1_aligned.png' '/face/hat/advhat/r100.pb'


#python3 face_preparation_my.py '/art/adversarial-robustness-toolbox/examples/dataset/john_hat.jpg' --mask


python3 cos_tf_simple.py --face1 '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned.png'\
        --face2 '/face/hat/advhat/Attack/rec_logo0.png'\
        --face3 '/face/hat/advhat/Attack/rec_logo1.png'\
        --face4 '/face/hat/advhat/Attack/rec_logo2.png'\
        --face5 '/face/hat/advhat/Attack/rec_logo3.png'\
        --model '/face/hat/advhat/r100.pb'



#python3 cos_tf_my.py --face1 '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned.png'\
        --face2 '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_withlogo1.png'\
        --face3 '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_withlogo2.png'\
        --face4 '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_withlogo3.png'\
        --face5 '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_withlogo4.png'\
        --model '/face/hat/advhat/r100.pb'


