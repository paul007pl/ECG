import torch.nn as nn
import torch
import sys
import os
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(proj_dir, "utils/emd"))
import emd_module as emd
sys.path.append(os.path.join(proj_dir, "utils/ChamferDistancePytorch"))
from chamfer3D import dist_chamfer_3D
chamLoss = dist_chamfer_3D.chamfer_3DDist()
sys.path.append(os.path.join(proj_dir, "utils/Pointnet2.PyTorch/pointnet2"))
import pointnet2_utils as pn2
from models.model_utils import get_uniform_loss


class FullModelECG(nn.Module):
    def __init__(self, model):
        super(FullModelECG, self).__init__()
        self.model = model
        self.EMD = emd.emdModule()

    def forward(self, inputs, gt, eps, iters, EMD=True, CD=True):
        cur_bs = inputs.size()[0]
        output1, output2 = self.model(inputs)
        gt = gt[:, :, :3]

        emd1 = emd2 = cd_p1 = cd_p2 = cd_t1 = cd_t2 = torch.tensor([0], dtype=torch.float32).cuda()

        if EMD:
            num_coarse = self.model.num_coarse
            gt_fps = pn2.gather_operation(gt.transpose(1, 2).contiguous(),
                                          pn2.furthest_point_sample(gt, num_coarse)).transpose(1, 2).contiguous()

            dist1, _ = self.EMD(output1, gt_fps, eps, iters)
            emd1 = torch.sqrt(dist1).mean(1)

            dist2, _ = self.EMD(output2, gt, eps, iters)
            emd2 = torch.sqrt(dist2).mean(1)

            # CD loss
        if CD:
            dist11, dist12, _, _ = chamLoss(gt, output1)
            cd_p1 = (torch.sqrt(dist11).mean(1) + torch.sqrt(dist12).mean(1)) / 2
            cd_t1 = (dist11.mean(1) + dist12.mean(1))

            dist21, dist22, _, _ = chamLoss(gt, output2)
            cd_p2 = (torch.sqrt(dist21).mean(1) + torch.sqrt(dist22).mean(1)) / 2
            cd_t2 = (dist21.mean(1) + dist22.mean(1))

        u1 = get_uniform_loss(output1)
        u2 = get_uniform_loss(output2)

        return output1, output2, emd1, emd2, cd_p1, cd_p2, cd_t1, cd_t2, u1, u2
