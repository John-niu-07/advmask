python main.py --edit_one_image            \
               --config celeba.yml         \
               --exp ./runs/test           \
               --t_0 500                   \
               --n_inv_step 40             \
               --n_test_step 40            \
               --n_iter 1                  \
               --img_path imgs/celeb1.png  \
               --model_path  checkpoint/test_FT_CelebA_HQ_neanderthal_t500_ninv40_ngen6_id0.0_l11.0_lr8e-06_Neanderthal-0.pth
               #--model_path  checkpoint/neanderthal.pth
