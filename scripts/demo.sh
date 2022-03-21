export GPU=0
export MODEL='pnp_soft-web-car_r50_90.1132.pth'
export DATASET='web-car'
export NCLASSES=196


python demo.py --model_path ${MODEL} --dataset ${DATASET} --nclasses ${NCLASSES} --gpu ${GPU}