export GPU=0


python main.py --config config/car.cfg --gpu ${GPU} --use-fp16 \
               --net resnet50 --batch-size 16 --lr 0.005 --opt sgd --warmup-lr-scale 1 --warmup-epochs 5 --epochs 120 --epsilon 0.5 \
               --lr-decay cosine \
               --alpha 1.0 --beta 1.0 --gamma 1.0 --omega 1.0 \
               --loss-func-aux mae --activation relu --classifier mlp-1 \
               --log pnp_soft --weighting soft