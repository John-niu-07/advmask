python main.py --clip_finetune          \
               --config celeba.yml      \
               --exp ./runs/test        \
               --edit_attr angry        \
               --do_train 1             \
               --do_test 1              \
               --n_train_img 10         \
               --n_test_img 5           \
               --n_iter 15              \
               --t_0 500                \
               --n_inv_step 40          \
               --n_train_step 6         \
               --n_test_step 40         \
               --lr_clip_finetune 8e-6  \
               --id_loss_w 1            \
               --l1_loss_w 1            