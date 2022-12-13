#python3 face_preparation.py '/art/adversarial-robustness-toolbox/examples/dataset/img2.jpg'
#python3 face_preparation.py '/art/adversarial-robustness-toolbox/examples/dataset/j4.jpg'


#python3 face_preparation.py '/art/adversarial-robustness-toolbox/examples/dataset/img32.jpg' --mask
#python3 face_preparation_recon.py '/art/adversarial-robustness-toolbox/examples/dataset/john_hat.jpg' --mask


#python3 attack.py '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned.png'  '/face/hat/advhat/r100.pb' --anchor_face '/art/adversarial-robustness-toolbox/examples/dataset/img32_aligned.png'
#python3 attack.py '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned.png'  '/face/hat/advhat/r100.pb' --anchor_face '/art/adversarial-robustness-toolbox/examples/dataset/img32_aligned.png'  --init_logo '/face/hat/advhat/Attack/example.png'


#python3 face_preparation_add.py '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned.png' --mask

python3 cos_tf_my.py --face1 '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned.png'\
       	--face2 '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_withlogo1.png'\
       	--face3 '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_withlogo2.png'\
       	--face4 '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_withlogo3.png'\
       	--face5 '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_rec.png'\
      	--model '/face/hat/advhat/r100.pb'




#python3 attack_add.py '/art/adversarial-robustness-toolbox/examples/dataset/img32_aligned.png' '/face/hat/advhat/r100.pb' --hat_face '/art/adversarial-robustness-toolbox/examples/dataset/img18_mask_mask.png'  --target_face '/art/adversarial-robustness-toolbox/examples/dataset/img32_aligned.png' --orig_face '/art/adversarial-robustness-toolbox/examples/dataset/img18_aligned.png'




