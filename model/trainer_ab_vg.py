import torch
import torch.nn as nn
from dataset.LVDataset import LVDataset
from model.model_ab_vg import NetABVG
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from model.utils import plot, save_loss, save_loss10
from model.data_utils import get_dataloader
import argparse
import time
import pickle
from AutomaticWeightedLoss.AutomaticWeightedLoss import AutomaticWeightedLoss

mp.set_start_method('spawn', force=True)
VISION_LOCAL_THRESHOLD = 1000.0

def train(model: NetABVG, train_loader, val_loader, params, device):
    model.to(device)
    model_id = params['model_id']
    awl = AutomaticWeightedLoss(params['n_awl'])
    optimizer, scheduler = model.get_optimizer(lr=params['lr'], steps_per_epoch=len(train_loader), num_epochs=params['epochs'], awl=awl)
    train_losses_global = []
    train_losses_local_pos = []
    train_losses_local_neg = []
    train_losses_local_pos1 = []
    train_losses_local_neg1 = []
    test_losses_global = []
    test_losses_local_pos = []
    test_losses_local_neg = []
    test_losses_local_pos1 = []
    test_losses_local_neg1 = []
    for epoch in range(params['epochs']):
        model.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            language, point_cloud, vision_global, vision_local, grasp_map, pos_index, neg_index, pos_neg_num, entry_paths = data
            language = language.to(device).float()
            point_cloud = point_cloud.to(device).float()
            vision_global = vision_global.to(device).float()
            vision_local = vision_local.to(device).float()
            grasp_map = grasp_map.to(device).float()
            pos_index = pos_index.to(device).long()
            neg_index = neg_index.to(device).long()
            pos_neg_num = pos_neg_num.to(device).long()
            
            vision_local_mean = torch.mean(vision_local, dim=(1,2))
            if torch.any(vision_local_mean > VISION_LOCAL_THRESHOLD):
                index = torch.nonzero(vision_local_mean > VISION_LOCAL_THRESHOLD)
                for i in index:
                    print("invalid data entry path", entry_paths[i])
                    print("vision_local mean", vision_local_mean[i])
                    print("vision_local", vision_local[i])
                continue
                
            output_global, output_local = model(language, point_cloud, vision_global, vision_local)
            loss_global, loss_local_pos, loss_local_neg, loss_local_pos1, loss_local_neg1 = model.get_loss(output_global, output_local, grasp_map, pos_index, neg_index, pos_neg_num, params['delta_v'], params['delta_d'])

            print(f'Train Epoch {epoch}, Batch {i}, Loss Global {loss_global.item()}, Loss Local Pos {loss_local_pos.item()}, Loss Local Neg {loss_local_neg.item()}, Loss Local Pos1 {loss_local_pos1.item()}, Loss Local Neg1 {loss_local_neg1.item()}')
            if params['n_awl'] == 2:
                loss = awl(loss_global, loss_local_pos * params['lambda_pos'] + loss_local_neg)
            else:
                assert params['n_awl'] == 3
                loss = awl(loss_global, loss_local_pos + loss_local_neg, loss_local_pos1 * params['lambda_pos'] + loss_local_neg1)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_losses_global.append(loss_global.item())
            train_losses_local_pos.append(loss_local_pos.item())
            train_losses_local_neg.append(loss_local_neg.item())
            train_losses_local_pos1.append(loss_local_pos1.item())
            train_losses_local_neg1.append(loss_local_neg1.item())
            

        test_loss_global, test_loss_local_pos, test_loss_local_neg, test_loss_local_pos1, test_loss_local_neg1 = validate(model, val_loader, device)

        test_losses_global += test_loss_global
        test_losses_local_pos += test_loss_local_pos
        test_losses_local_neg += test_loss_local_neg
        test_losses_local_pos1 += test_loss_local_pos1
        test_losses_local_neg1 += test_loss_local_neg1
        # save_loss(train_losses_global=train_losses_global, train_losses_local_pos=train_losses_local_pos, train_losses_local_neg=train_losses_local_neg, test_losses_global=test_losses_global, test_losses_local_pos=test_losses_local_pos, test_losses_local_neg=test_losses_local_neg, model_id=model_id)
        save_loss10(train_losses_global=train_losses_global, train_losses_local_pos=train_losses_local_pos, train_losses_local_neg=train_losses_local_neg, train_losses_local_pos1=train_losses_local_pos1, train_losses_local_neg1=train_losses_local_neg1, test_losses_global=test_losses_global, test_losses_local_pos=test_losses_local_pos, test_losses_local_neg=test_losses_local_neg, test_losses_local_pos1=test_losses_local_pos1, test_losses_local_neg1=test_losses_local_neg1, model_id=model_id)
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'awl': awl.state_dict(),
            'epoch': epoch
        }
        checkpoint_path = './checkpoints/model_{}_{}.pth'.format(model_id, epoch)
        torch.save(checkpoint, checkpoint_path)
        
    return train_losses_global, train_losses_local_pos, train_losses_local_neg, test_losses_global, test_losses_local_pos, test_losses_local_neg

def validate(model: NetABVG, val_loader, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        test_losses_global = []
        test_losses_local_pos = []
        test_losses_local_neg = []
        test_losses_local_pos1 = []
        test_losses_local_neg1 = []
        for i, data in enumerate(val_loader):
            language, point_cloud, vision_global, vision_local, grasp_map, pos_index, neg_index, pos_neg_num, entry_paths = data
            language = language.to(device).float()
            point_cloud = point_cloud.to(device).float()
            vision_global = vision_global.to(device).float()
            vision_local = vision_local.to(device).float()
            grasp_map = grasp_map.to(device).float()
            pos_index = pos_index.to(device).long()
            neg_index = neg_index.to(device).long()
            pos_neg_num = pos_neg_num.to(device).long()
            vision_local_mean = torch.mean(vision_local, dim=(1,2))
            if torch.any(vision_local_mean > VISION_LOCAL_THRESHOLD):
                index = torch.nonzero(vision_local_mean > VISION_LOCAL_THRESHOLD)
                for i in index:
                    print("invalid data entry path", entry_paths[i])
                    print("vision_local mean", vision_local_mean[i])
                    print("vision_local", vision_local[i])
                continue
            
            output_global, output_local = model(language, point_cloud, vision_global, vision_local)
            loss_global, loss_local_pos, loss_local_neg, loss_local_pos1, loss_local_neg1 = model.get_loss(output_global, output_local, grasp_map, pos_index, neg_index, pos_neg_num, params['delta_v'], params['delta_d'])

            test_losses_global.append(loss_global.item())
            test_losses_local_pos.append(loss_local_pos.item())
            test_losses_local_neg.append(loss_local_neg.item())
            test_losses_local_pos1.append(loss_local_pos1.item())
            test_losses_local_neg1.append(loss_local_neg1.item())
            print(f'Validate Batch {i}, Loss Global {loss_global.item()}, Loss Local Pos {loss_local_pos.item()}, Loss Local Neg {loss_local_neg.item()}, Loss Local Pos1 {loss_local_pos1.item()}, Loss Local Neg1 {loss_local_neg1.item()}')
    
    return test_losses_global, test_losses_local_pos, test_losses_local_neg, test_losses_local_pos1, test_losses_local_neg1

    
def main(params):
    train_loader, val_loader, test_loader = get_dataloader(params)
    model = NetABVG()
    device = torch.device('cuda:0')
    train_losses_global, train_losses_local_pos, train_losses_local_neg, test_losses_global, test_losses_local_pos, test_losses_local_neg = train(model, train_loader, val_loader, params, device)
    print("finished training")
    validate(model, test_loader, device)
    # plot(train_losses_global, train_losses_local_pos, train_losses_local_neg, test_losses_global, test_losses_local_pos, test_losses_local_neg, model_id=params['model_id'])

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset_dir', type=str, default='./data/objects/')
    argparser.add_argument('--shuffle', type=bool, default=True)
    argparser.add_argument('--epochs', type=int, default=40)
    argparser.add_argument('--batch_size', type=int, default=128)
    argparser.add_argument('--num_workers', type=int, default=16)
    argparser.add_argument('--lr', type=float, default=2e-3)
    argparser.add_argument('--small_data', default=False, action='store_true')
    argparser.add_argument('--delta_v', type=float, default=0.5)
    argparser.add_argument('--delta_d', type=float, default=3.0)
    argparser.add_argument('--lambda_pos', type=float, default=1.0)
    argparser.add_argument('--n_awl', type=int, default=2)
    args = argparser.parse_args()
    params = vars(args)
    params['model_id'] = time.time()
    main(params)