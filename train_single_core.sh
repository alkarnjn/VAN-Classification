# MODEL=van_b0 # van_{b0, b1, b2, b3}
# DROP_PATH=0.1 # drop path rates [0.1, 0.1, 0.1, 0.2] for [b0, b1, b2, b3]
# CUDA_VISIBLE_DEVICES=0 bash distributed_train.sh 1 imagenet \
# 	  --model van_b0 -b 128 --lr 1e-3 --drop-path 0.1


python train.py  "D:/Others/fvc2002_out_splitted" --model van_b1 -b 128 --lr 1e-3 --drop-path 0.1 --num-classes 110 --epochs 200

python validate.py "D:/Others/fvc2002_out_splitted/test" --model van_b1 -b 128 --num-classes 110

