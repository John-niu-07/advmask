rm -rf ./checkpoint/*.*

python main_adv.py --clip_finetune          \
               --config celeba.yml      \
               --exp ./runs/test        \
               --edit_attr beards         \
               --do_train 1             \
               --do_test 0              \
               --n_train_img 2          \
               --n_test_img 1           \
               --n_iter 15              \
               --t_0 500                \
               --n_inv_step 40          \
               --n_train_step 4        \
               --n_test_step 40         \
               --lr_clip_finetune 8e-6  \
               --id_loss_w 1            \
               --l1_loss_w 0            \
	       --clip_loss_w 0		\
	       --n_precomp_img 2
