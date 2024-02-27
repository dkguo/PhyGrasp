import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR


class NetABVL(nn.Module):
    def __init__(self, language_size = 4096, vision_global_size = 1024, vision_local_size = 64,):
        super(NetABVL, self).__init__() 
        # input
        self.language_size = language_size
        self.vision_global_size = vision_global_size
        self.vision_local_size = vision_local_size

        # feature size
        self.language_feature_size = 128
        self.vision_feature_global_size = 128
        self.global_feature_size = 64
        self.global_local_size = 64 + 3

        # point cloud
        self.num_points = 2048

        # output size
        self.output_global_size = 1
        self.output_local_size = 32
        
        self.language_feature = torch.nn.Sequential(
            torch.nn.LayerNorm(self.language_size), 
            torch.nn.Linear(self.language_size, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.language_feature_size)
        )

        self.vision_global = torch.nn.Sequential(
            torch.nn.LayerNorm(self.vision_global_size),
            torch.nn.Linear(self.vision_global_size, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.vision_feature_global_size)
        )

        self.global_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.language_feature_size + self.vision_feature_global_size, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.global_feature_size)
        )

        self.point_cloud_norm = torch.nn.LayerNorm(3)
        self.vision_local_norm = torch.nn.LayerNorm(vision_local_size)
        
        # self.atten = nn.MultiheadAttention(embed_dim=self.global_local_size, num_heads=1)
            
        self.out_mlp1 = torch.nn.Sequential(
            torch.nn.Linear(self.global_local_size, 256), # (batch_size, 2048, 256)
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.output_global_size)
        )

        self.out_mlp2 = torch.nn.Sequential(
            torch.nn.Linear(self.global_local_size, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.output_local_size)
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.output_local_size * 2, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )

    def forward(self, language, point_cloud, vision_global, vision_local):
        '''
        language: (batch_size, 4096)
        point_cloud: (batch_size, 2048, 3)
        vision_global: (batch_size, 1024)
        vision_local: (batch_size, 2048, 64)
        '''
        # torch.set_printoptions(threshold=100000)
        # print(language.shape, point_cloud.shape, vision_global.shape, vision_local.shape)
        # print(language.dtype, point_cloud.dtype, vision_global.dtype, vision_local.dtype)

        language_feature = self.language_feature(language) # (batch_size, 128)
        vision_global_feature = self.vision_global(vision_global) # (batch_size, 128)
        global_feature = torch.cat((language_feature, vision_global_feature), dim=1) # (batch_size, 256)
        # MLP 256->64
        global_feature = self.global_mlp(global_feature) # (batch_size, 64)
        global_feature = global_feature.unsqueeze(1).repeat(1, self.num_points, 1) # (batch_size, 2048, 64)
        # vision_local = self.vision_local_norm(vision_local) # (batch_size, 2048, 64)
        point_cloud = self.point_cloud_norm(point_cloud) # (batch_size, 2048, 3)
        # print("vision_local", vision_local)
        # print("point_cloud", point_cloud)

        global_local = torch.cat((global_feature, point_cloud), dim=2) # (batch_size, 2048, 64+3)
        

        # print("global_local", global_local)
        # global_local = global_local.permute(1, 0, 2) # (2048, batch_size, 64+64+3)
        # global_local, _ = self.atten(global_local, global_local, global_local) # (2048, batch_size,  64+64+3)
        # global_local = global_local.permute(1, 0, 2) # (batch_size, 2048, 64+64+3)
        # print("global_local", global_local)

        output_global = self.out_mlp1(global_local) # (batch_size, 2048, 1)
        output_local = self.out_mlp2(global_local) # (batch_size, 2048, 32)
        # print("output_global", output_global)
        # print("output_local", output_local)

        return output_global, output_local

    def get_optimizer(self, lr=0.001, steps_per_epoch=100, num_epochs=100, awl=None):

        optimizer = torch.optim.Adam([{'params': self.parameters()},
                                     {'params': awl.parameters(), 'weight_decay': 0}],
                                     lr=lr)
        scheduler = OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=num_epochs)
        return optimizer, scheduler


    def get_loss(self, output_global, output_local, grasp_map, pos_index, neg_index, pos_neg_num, delta_v, delta_d):
        '''
        output_global: (batch_size, 2048, 1)
        output_local: (batch_size, 2048, 32)
        grasp_map: (batch_size, 2048)
        pos_index: (batch_size, 200, 2), pos_index gives the index of pairs of two points that distance should be minimized
        neg_index: (batch_size, 200, 2), neg_index gives the index of pairs of two points that distance should be maximized
        pos_neg_num: (batch_size, 2), pos_neg_num[0] is the number of positive pairs, pos_neg_num[1] is the number of negative pairs
        
        '''
        # print(output_global.dtype, output_local.dtype, grasp_map.dtype, pos_index.dtype, neg_index.dtype)
        # torch.set_printoptions(threshold=100000)
        # print("pos_neg_num", pos_neg_num)
        # print("pos_index", pos_index)
        # print("neg_index", neg_index)
        grasp_map = grasp_map.unsqueeze(2) # (batch_size, 2048, 1)
        loss_global = torch.nn.functional.mse_loss(output_global, grasp_map)

        pos_index_0 = pos_index[:, :, 0].unsqueeze(2).expand(-1, -1, self.output_local_size) # (batch_size, 200, 32)
        pos_index_1 = pos_index[:, :, 1].unsqueeze(2).expand(-1, -1, self.output_local_size) # (batch_size, 200, 32)
        neg_index_0 = neg_index[:, :, 0].unsqueeze(2).expand(-1, -1, self.output_local_size) # (batch_size, 200, 32)
        neg_index_1 = neg_index[:, :, 1].unsqueeze(2).expand(-1, -1, self.output_local_size) # (batch_size, 200, 32)

        pos_0 = torch.gather(output_local, 1, pos_index_0) # (batch_size, 200, 32), pos_0[i][j][k] = output_local[i][pos_index[i][j][k]][k]
        pos_1 = torch.gather(output_local, 1, pos_index_1) # (batch_size, 200, 32)
        neg_0 = torch.gather(output_local, 1, neg_index_0) # (batch_size, 200, 32)
        neg_1 = torch.gather(output_local, 1, neg_index_1) # (batch_size, 200, 32)

        pos = torch.cat((pos_0, pos_1), dim=2) # (batch_size, 200, 64)
        neg = torch.cat((neg_0, neg_1), dim=2) # (batch_size, 200, 64)

        pos_score = self.classifier(pos) # (batch_size, 200, 1)
        neg_score = self.classifier(neg) # (batch_size, 200, 1)

        pos_score = pos_score.squeeze(2) # (batch_size, 200)
        neg_score = neg_score.squeeze(2) # (batch_size, 200)

        # loss_local_pos = 1 - torch.nn.functional.cosine_similarity(pos_0, pos_1, dim=2) # (batch_size, 200)
        # loss_local_neg = 1 + torch.nn.functional.cosine_similarity(neg_0, neg_1, dim=2) # (batch_size, 200)
        loss_local_pos1 = torch.clamp(torch.norm(pos_0 - pos_1, dim=2) - delta_v, min=0.0) # (batch_size, 200)
        loss_local_neg1 = torch.clamp(delta_d - torch.norm(neg_0 - neg_1, dim=2), min=0.0) # (batch_size, 200)
        # loss_local_pos = 1.0 - torch.abs(torch.nn.functional.cosine_similarity(pos_0, pos_1, dim=2)) # (batch_size, 200)
        # loss_local_neg = torch.abs(torch.nn.functional.cosine_similarity(neg_0, neg_1, dim=2)) # (batch_size, 200)
        # loss_local_pos = torch.clamp(loss_local_pos - delta_v, min=0.0) # (batch_size, 200)
        # loss_local_neg = torch.clamp(loss_local_neg - delta_d, min=0.0) # (batch_size, 200)
        # print("pos_score", pos_score)
        # print("neg_score", neg_score)

        loss_local_pos = torch.nn.BCEWithLogitsLoss(reduction='none')(pos_score, torch.ones_like(pos_score)) # (batch_size, 200)
        loss_local_neg = torch.nn.BCEWithLogitsLoss(reduction='none')(neg_score, torch.zeros_like(neg_score)) # (batch_size, 200)

        range_tensor = torch.arange(0, loss_local_pos.shape[1]).unsqueeze(0).expand_as(loss_local_pos).to(loss_local_pos.device) # (batch_size, 200)
        mask_pos = (range_tensor < pos_neg_num[:, 0].unsqueeze(1)) # (batch_size, 200)
        mask_neg = (range_tensor < pos_neg_num[:, 1].unsqueeze(1)) # (batch_size, 200) 
        # print("mask_pos", mask_pos)
        # print("mask_neg", mask_neg)

        loss_local_pos = (loss_local_pos[mask_pos]).mean()
        loss_local_neg = (loss_local_neg[mask_neg]).mean()
        
        # loss_local_pos = (loss_local_pos[mask_pos]).mean()
        # loss_local_neg = (loss_local_neg[mask_neg]).mean()
        loss_local_pos1 = (loss_local_pos1[mask_pos] ** 2).mean()
        loss_local_neg1 = (loss_local_neg1[mask_neg] ** 2).mean()

        # check balance with log
        return loss_global, loss_local_pos, loss_local_neg, loss_local_pos1, loss_local_neg1
    
    def get_score(self, output_local, index1):
        '''
        output_local: (batch_size, 2048, 32)
        index: (batch_size, kp1)
        '''
        # index = index.view(-1, 1, 1).expand(-1, -1, output_local.shape[2]) # (batch_size, 1, 32)
        # p0 = torch.gather(output_local, 1, index) # (batch_size, 1, 32)
        # p0 = p0.expand(-1, output_local.shape[1], -1) # (batch_size, 2048, 32)
        # p1 = output_local # (batch_size, 2048, 32)
        # p = torch.cat((p0, p1), dim=2) # (batch_size, 2048, 64)
        # score = self.classifier(p) # (batch_size, 2048, 1)
        # score = score.squeeze(2) # (batch_size, 2048)
        index_1 = index1.unsqueeze(2).expand(-1, -1, output_local.shape[2]) # (batch_size, kp1, 32)
        p0 = torch.gather(output_local, 1, index_1) # (batch_size, kp1, 32)
        p0 = p0.unsqueeze(2).expand(-1, -1, output_local.shape[1], -1) # (batch_size, kp1, 2048, 32)
        p1 = output_local.unsqueeze(1).expand(-1, index1.shape[1], -1, -1) # (batch_size, kp1, 2048, 32)
        p = torch.cat((p0, p1), dim=3) # (batch_size, kp1, 2048, 64)
        score = self.classifier(p) # (batch_size, kp1, 2048, 1)
        return score
    
    def get_pos(self, output_local, index1, point_cloud):
        '''
        output_local: (batch_size, 2048, 32)
        index1: (batch_size, kp1)
        point_cloud: (batch_size, 2048, 3)
        return: pos: (batch_size, kp1, 6)
        '''
        index_1 = index1.unsqueeze(2).expand(-1, -1, output_local.shape[2]) # (batch_size, kp1, 32)
        p0 = torch.gather(output_local, 1, index_1) # (batch_size, kp1, 32)
        p0 = p0.unsqueeze(2).expand(-1, -1, point_cloud.shape[1], -1) # (batch_size, kp1, 2048, 32)
        p1 = output_local.unsqueeze(1).expand(-1, index1.shape[1], -1, -1) # (batch_size, kp1, 2048, 32)
        p = torch.cat((p0, p1), dim=3) # (batch_size, kp1, 2048, 64)
        score = self.classifier(p) # (batch_size, kp1, 2048, 1)
        score = score.squeeze(3) # (batch_size, kp1, 2048)
        index2 = torch.argmax(score, dim=2) # (batch_size, kp1)
        index_1 = index2.unsqueeze(2).expand(-1, -1, point_cloud.shape[2]) # (batch_size, kp1, 3)
        pos1 = torch.gather(point_cloud, 1, index_1) # (batch_size, kp1, 3)
        index_2 = index1.unsqueeze(2).expand(-1, -1, point_cloud.shape[2]) # (batch_size, kp1, 3)
        pos2 = torch.gather(point_cloud, 1, index_2) # (batch_size, kp1, 3)
        pos = torch.cat((pos1, pos2), dim=2) # (batch_size, kp1, 6)
        return pos
        
        
