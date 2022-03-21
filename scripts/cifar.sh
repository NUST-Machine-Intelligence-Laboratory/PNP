export GPU=0


python cifar.py --config config/cifar100.cfg --synthetic-data cifar80no --noise-type asymmetric --closeset-ratio 0.4 \
                --gpu ${GPU} --use-fp16 \
                --net cnn --batch-size 128 --lr 0.001 --opt adam --warmup-lr-scale 1 --warmup-epochs 10 --epochs 200 --epsilon 0.6 \
                --lr-decay linear --loss-func-aux mae --activation relu --temperature 0.1 --eta 0.5 \
                --alpha 0.0 --beta 1.0 --gamma 1.0 --omega 1.0 \
                --log pnp_soft --weighting soft