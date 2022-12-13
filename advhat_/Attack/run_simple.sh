

#python3 face_preparation_add.py '/art/adversarial-robustness-toolbox/examples/dataset/john_hat.jpg' --mask

#untargeted
#python3 attack_simple.py '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned.png'  '/face/hat/advhat/r100.pb' 
#python3 attack_simple.py '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned.png'  '/face/hat/advhat/r100.pb'  --init_log '/face/hat/advhat/Attack/hat_out2.png'

#targeted
#python3 attack_rect.py '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned.png'  '/face/hat/advhat/r100.pb' --anchor_face '/art/adversarial-robustness-toolbox/examples/dataset/jj3_aligned.png' --init_log '/face/hat/advhat/Attack/john_hatcrop.png'
#python3 attack_rect.py '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned.png'  '/face/hat/advhat/r100.pb' --anchor_face '/art/adversarial-robustness-toolbox/examples/dataset/jj3_aligned.png'
#python3 attack.py '/art/adversarial-robustness-toolbox/examples/dataset/mat_hat_aligned.png'  '/face/hat/advhat/r100.pb' 



#python3 attack_simple_rec.py '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned.png'  '/face/hat/advhat/r100.pb'  --logo0 '/face/hat/advhat/Attack/logo_0.png' --logo1 '/face/hat/advhat/Attack/logo_20.png' \
	--logo2 '/face/hat/advhat/Attack/logo_40.png' --logo3 '/face/hat/advhat/Attack/logo_60.png'



#python3 cos_tf_simple.py --face1 '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned.png'\
        --face2 '/face/hat/advhat/Attack/rec_logo0.png'\
        --face3 '/face/hat/advhat/Attack/rec_logo1.png'\
        --face4 '/face/hat/advhat/Attack/rec_logo2.png'\
        --face5 '/face/hat/advhat/Attack/rec_logo3.png'\
        --model '/face/hat/advhat/r100.pb'




