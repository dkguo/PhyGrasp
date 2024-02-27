import os
if __name__ == '__main__':
    # n_awl: 2: w/o embedding loss, 3: w/ embedding loss
    os.system("python -u -m model.trainer       --lr 1e-3 --n_awl 3 --delta_v 0.4 --delta_d 2.0 --lambda_p 2.0")
    # os.system("python -u -m model.trainer       --lr 1e-3 --n_awl 2 --delta_v 0.4 --delta_d 2.0 --lambda_p 2.0")
    os.system("python -u -m model.trainer_ab_l  --lr 1e-3 --n_awl 3 --delta_v 0.4 --delta_d 2.0 --lambda_p 2.0")
    os.system("python -u -m model.trainer_ab_vg --lr 1e-3 --n_awl 3 --delta_v 0.4 --delta_d 2.0 --lambda_p 2.0")
    os.system("python -u -m model.trainer_ab_vl --lr 1e-3 --n_awl 3 --delta_v 0.4 --delta_d 2.0 --lambda_p 2.0")

