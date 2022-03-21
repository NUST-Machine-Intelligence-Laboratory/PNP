export GPU=0


python main.py --config config/bird.cfg --gpu ${GPU} --use-fp16 \
               --net resnet50 --batch-size 16 --lr 0.0005 --opt sgd --warmup-lr-scale 10 --warmup-epochs 5 --epochs 120 --epsilon 0.35 \
               --lr-decay cosine \
               --alpha 1.0 --beta 1.0 --gamma 1.0 --omega 1.0 \
               --loss-func-aux mae --activation tanh --classifier linear \
               --log pnp_soft --weighting soft