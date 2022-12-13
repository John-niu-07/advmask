

#python3 face_preparation_add.py '/art/adversarial-robustness-toolbox/examples/dataset/john_hat.jpg' --mask

#untargeted
#python3 attack_rect.py '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned.png'  '/face/hat/advhat/r100.pb'  --init_log '/face/hat/advhat/Attack/john_hatcrop.png'

#targeted
python3 attack2_rect.py '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned.png'  '/face/hat/advhat/r100.pb' --anchor_face '/art/adversarial-robustness-toolbox/examples/dataset/img2_aligned.png' --init_log '/face/hat/advhat/Attack/john_hatcrop2.png'
#python3 attack_rect.py '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned.png'  '/face/hat/advhat/r100.pb' --anchor_face '/art/adversarial-robustness-toolbox/examples/dataset/jj3_aligned.png'
#python3 attack.py '/art/adversarial-robustness-toolbox/examples/dataset/mat_hat_aligned.png'  '/face/hat/advhat/r100.pb' 





#python3 cos_tf_add.py



#python3 cos_tf_add.py '/art/adversarial-robustness-toolbox/examples/dataset/img2_aligned.png'\
        '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned.png'\
        '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned_mask.png'\
        '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned_mask_eg.png'\
        '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned_mask_init.png'\
        '/face/hat/advhat/r100.pb'

