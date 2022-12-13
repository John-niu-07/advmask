

#python3 face_preparation_add.py '/art/adversarial-robustness-toolbox/examples/dataset/john_hat.jpg' --mask

#untargeted
#python3 attack_simple112.py '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned112.png'  '/face/hat/advhat/r100.pb' 
#python3 attack_simple112.py '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned112.png'  '/face/hat/advhat/r100.pb'  --init_log '/face/hat/advhat/Attack/hat_out112.png'

#targeted
#python3 attack_simple112.py '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned112.png'  '/face/hat/advhat/r100.pb' --anchor_face '/art/adversarial-robustness-toolbox/examples/dataset/jj3_aligned.png' --init_log '/face/hat/advhat/Attack/hat_out112.png'
#python3 attack_simple112_full.py '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned112.png'  '/face/hat/advhat/r100.pb' --anchor_face '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned112.png' --init_log '/face/hat/advhat/Attack/john_init.png'







python3 cos_tf_simple112_full.py 





