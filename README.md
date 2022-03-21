Introduction
---
The source code and models for our paper **PNP: Robust Learning from Noisy Labels by Probabilistic Noise Prediction**


Framework
---
![framework](asserts/framework.jpg)


Installation
---
After creating a virtual environment of python 3.6, run `pip install -r requirements.txt` to install all dependencies


How to use
---
The code is currently tested only on GPU.

- Data preparation

    Created a folder `Datasets` and download `cifar100`/`web-aircraft`/`web-bird`/`web-car`/`food101n` dataset into this folder.


- Source code
    - If you want to train the whole model from beginning using the source code, please follow subsequent steps:
        - Prepare data
        - Modify GPU device in the corresponding train script `xxx.sh` in `scripts` folder
        - Activate virtual environment (e.g. conda) and then run
        ```
        bash scripts/xxx.sh
        ```


- Demo
    - If you just want to do a quick test on the model, please follow subsequent steps:
      - Prepare data
      - Download one of the following trained models
        ```
        wget https://web-pnp.oss-cn-shanghai.aliyuncs.com/pnp_hard-f101n-r50-87.3109.pth
        wget https://web-pnp.oss-cn-shanghai.aliyuncs.com/pnp_soft-f101n-r50-87.5010.pth
        wget https://web-pnp.oss-cn-shanghai.aliyuncs.com/pnp_hard-web_car-r50_89.9266.pth
        wget https://web-pnp.oss-cn-shanghai.aliyuncs.com/pnp_soft-web-car_r50_90.1132.pth
        ```
      - Modify `GPU`, `MODEL`, `DATASET`, and `NCLASSES` accordingly in the demo script `demo.sh` in `scripts` folder
      - Activate virtual environment (e.g. conda) and then run
        ```
        bash scripts/demo.sh
        ```