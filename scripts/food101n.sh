export GPU=0


python main.py --config config/food101n.cfg --gpu ${GPU} --use-fp16 \
               --net resnet50 --batch-size 96 --lr 0.01 --opt sgd --weight-decay 5e-4 --warmup-lr-scale 1 --warmup-epochs 5 --epochs 50 --epsilon 0.5 \
               --lr-decay step \
               --alpha 0.75 --beta 0.3 --gamma 1.0 --omega 0.2 \
               --loss-func-aux mae --activation relu --classifier mlp-1 \
               --log pnp_soft --weighting soft
