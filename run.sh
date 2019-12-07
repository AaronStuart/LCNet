conda activate pytorch-gpu

python -m visdom.server

python train.py --pretrained_weights /home/stuart/PycharmProjects/EDANet/weights/EDANet/epoch_0_iter_18000.pth