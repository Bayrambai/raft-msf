import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from .update import BasicUpdateBlock, conv
from .extractor import BasicEncoder
from .corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8, proj_coords_grid, upsample_x8
from utils.sceneflow_util import flow_horizontal_flip, intrinsic_scale, get_pixelgrid, post_processing
from utils.sceneflow_util import projectSceneFlow2Flow

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args
        self.sigmoid = torch.nn.Sigmoid()
        self.disp_head =  nn.Sequential(
            conv(256, 128),
            conv(128, 1, isReLU=False),
            )
        
        self.sf_head =  nn.Sequential(
            conv(256, 128),
            conv(128, 3, isReLU=False),
            )

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def cvx_up_pred(self, pred, mask):
        N, dim, H, W = pred.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)
        up_pred = F.unfold(pred, [3,3], padding=1)
        up_pred = up_pred.view(N, dim, 9, 1, 1, H, W)
        up_pred = torch.sum(mask * up_pred, dim=2)
        up_pred = up_pred.permute(0, 1, 4, 2, 5, 3)
        return up_pred.reshape(N, dim, 8*H, 8*W)

    def sceneF2opticalF(self, disp, sf, K, input_size):
        _, _, h, w = disp.size()
        disp = disp * w
        local_scale = torch.zeros_like(input_size)
        local_scale[:, 0] = h
        local_scale[:, 1] = w
        rel_scale = local_scale / input_size
        intrinsic_s = intrinsic_scale(K, rel_scale[:, 0], rel_scale[:, 1])
        proj_flow, coord = projectSceneFlow2Flow(intrinsic_s, sf, disp)
        return proj_flow, coord

    def run_raft(self, image1, image2, K, input_size):
        """ Estimate optical flow between pair of frames """
        iters = self.args.iters
        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        # RAFT SceneFlow
        b, _, h, w = image1.shape
        sf = torch.zeros(b, 3, h//8, w//8).to(image1.device)
        disp = self.disp_head(fmap1)
        

        coords0 = coords_grid(b, h//8, w//8).to(image1.device)
        _, coords1 = self.sceneF2opticalF(disp * 0.3, sf, K, input_size) # multiplied by 0.3: ijcvRevision

        disp_predictions = []
        sf_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            disp = disp.detach()
            sf = sf.detach()
            corr = corr_fn(coords1) # index correlation volume
            flow = coords1 - coords0

            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, disp_d, sf_d = self.update_block(net, inp, corr, flow, disp, sf)
            disp = disp + disp_d
            sf = sf + sf_d
            _, coords1 = self.sceneF2opticalF(self.sigmoid(disp) * 0.3, sf, K, input_size) 
            
            if up_mask is None:
                disp_up = upsample_x8(self.sigmoid(disp) * 0.3)
                sf_up = upsample_x8(sf)
            else:
                disp_up = self.cvx_up_pred(self.sigmoid(disp) * 0.3, up_mask)
                sf_up = self.cvx_up_pred(sf, up_mask)
            disp_predictions.append(disp_up)
            sf_predictions.append(sf_up)
        return disp_predictions, sf_predictions
    
    def forward(self, input_dict):
        
        output_dict = {}
        k1 = input_dict['input_k_l1_aug']
        k2 = input_dict['input_k_l2_aug']
        input_size = input_dict['aug_size']

        ## Left
        output_dict['disp_l1'], output_dict['flow_f'] = self.run_raft(input_dict['input_l1_aug'], input_dict['input_l2_aug'], k1, input_size)
        output_dict['disp_l2'], output_dict['flow_b'] = self.run_raft(input_dict['input_l2_aug'], input_dict['input_l1_aug'], k2, input_size)
        if self.training or (not self.args.finetuning and not self.args.evaluation):
            output_dict_r = {}
            input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
            input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
            k_r1_flip = input_dict["input_k_r1_flip_aug"]
            k_r2_flip = input_dict["input_k_r2_flip_aug"]

            disp1_pred, sf_f_pred = self.run_raft(input_r1_flip, input_r2_flip, k_r1_flip, input_size)
            disp2_pred, sf_b_pred = self.run_raft(input_r2_flip, input_r1_flip, k_r2_flip, input_size)

            output_dict_r['flow_f'] = [ flow_horizontal_flip(sf_f_pred[ii]) for ii in range(len(sf_f_pred)) ]
            output_dict_r['flow_b'] = [ flow_horizontal_flip(sf_b_pred[ii]) for ii in range(len(sf_b_pred)) ]
            output_dict_r['disp_l1'] = [ torch.flip(disp1_pred[ii], [3]) for ii in range(len(disp1_pred)) ]
            output_dict_r['disp_l2'] = [ torch.flip(disp2_pred[ii], [3]) for ii in range(len(disp2_pred)) ]

            output_dict['output_dict_r'] = output_dict_r

        ## Post Processing 
        ## ss:           eval
        ## ft: train val eval
        if self.args.evaluation or self.args.finetuning:

            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            disp1_pred, sf_f_pred = self.run_raft(input_l1_flip, input_l2_flip, k_l1_flip, input_size)
            disp2_pred, sf_b_pred = self.run_raft(input_l2_flip, input_l1_flip, k_l2_flip, input_size)

            flow_f_pp = []
            flow_b_pp = []
            disp_l1_pp = []
            disp_l2_pp = []

            for ii in range(0, len(disp1_pred)):

                flow_f_pp.append(post_processing(output_dict['flow_f'][ii], flow_horizontal_flip(sf_f_pred[ii])))
                flow_b_pp.append(post_processing(output_dict['flow_b'][ii], flow_horizontal_flip(sf_b_pred[ii])))
                disp_l1_pp.append(post_processing(output_dict['disp_l1'][ii], torch.flip(disp1_pred[ii], [3])))
                disp_l2_pp.append(post_processing(output_dict['disp_l2'][ii], torch.flip(disp2_pred[ii], [3])))

            output_dict['flow_f_pp'] = flow_f_pp
            output_dict['flow_b_pp'] = flow_b_pp
            output_dict['disp_l1_pp'] = disp_l1_pp
            output_dict['disp_l2_pp'] = disp_l2_pp
        return output_dict

