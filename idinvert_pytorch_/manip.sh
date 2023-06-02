MODEL_NAME='styleganinv_ffhq256'
IMAGE_DIR='results/inversion/my_test7_'
BOUNDARY='boundaries/stylegan_ffhq256/expression.npy'
python manipulate.py $MODEL_NAME $IMAGE_DIR $BOUNDARY
