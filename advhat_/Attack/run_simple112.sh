

#python3 face_preparation_add.py '/art/adversarial-robustness-toolbox/examples/dataset/john_hat.jpg' --mask

#untargeted
#python3 attack_simple112.py '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned112.png'  '/face/hat/advhat/r100.pb' 
#python3 attack_simple112.py '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned112.png'  '/face/hat/advhat/r100.pb'  --init_log '/face/hat/advhat/Attack/hat_out112.png'

#targeted
#python3 attack_simple112.py '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned112.png'  '/face/hat/advhat/r100.pb' --anchor_face '/art/adversarial-robustness-toolbox/examples/dataset/jj3_aligned.png' --init_log '/face/hat/advhat/Attack/hat_out112.png'
#python3 attack_simple112.py '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned112.png'  '/face/hat/advhat/r100.pb' --anchor_face '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned112.png' --init_log '/face/hat/advhat/Attack/logo112_init.png'
#python3 attack_simple112.py '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned112.png'  '/face/hat/advhat/r100.pb' --anchor_face '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned112.png'



#python tool_simple112.py


#python3 attack_simple112_rec.py '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned112.png'  '/face/hat/advhat/r100.pb'  --logo0 '/face/hat/advhat/Attack/logo112_0.png' --logo1 '/face/hat/advhat/Attack/logo112_20.png' \
#	--logo2 '/face/hat/advhat/Attack/logo112_40.png' --logo3 '/face/hat/advhat/Attack/logo112_60.png' --logo4 '/face/hat/advhat/Attack/logo112_80.png'




python3 cos_tf_simple112.py 

#python3 cos_tf_simple112.py --face1 '/art/adversarial-robustness-toolbox/examples/dataset/john_hat_aligned112.png'\
#        --face2 '/face/hat/advhat/Attack/rec112_logo0.png'\
#        --face3 '/face/hat/advhat/Attack/rec112_logo1.png'\
#        --face4 '/face/hat/advhat/Attack/rec112_logo2.png'\
#        --face5 '/face/hat/advhat/Attack/rec112_logo3.png'\
#        --model '/face/hat/advhat/r100.pb'




