MODEL=van_b0 # van_{b0, b1, b2, b3}
DROP_PATH=0.1 # drop path rates [0.1, 0.1, 0.1, 0.2] for [b0, b1, b2, b3]
CUDA_VISIBLE_DEVICES=0 bash distributed_train.sh 1 imagenet \
	  --model $MODEL -b 128 --lr 1e-3 --drop-path $DROP_PATH --num-classes 150 --epochs 50
