from models.ecg_model import ECG
from utils.utils import *
import torch
import os
import h5py
import sys
import os
proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(proj_dir, "utils/emd"))
import emd_module as emd
EMD = emd.emdModule()

sys.path.append(os.path.join(proj_dir, "utils/ChamferDistancePytorch"))
from chamfer3D import dist_chamfer_3D
from fscore import fscore
chamLoss = dist_chamfer_3D.chamfer_3DDist()


def test(args):
    model_dir = args.model_dir
    log_test = LogString(open(os.path.join(model_dir, 'log_text.txt'), 'w'))
    dataset_test = ShapeNetH5(train=False, npoints=args.num_points, use_mean_feature=args.use_mean_feature)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=int(args.workers))
    dataset_length = len(dataset_test)
    epochs = ['best_emd_network.pth', 'best_cd_p_network.pth', 'best_cd_t_network.pth']
    for epoch in epochs:
        load_path = os.path.join(args.model_dir, epoch)
        log_test.log_string(load_path)
        if args.model_name == 'ecg':
            net = ECG(num_coarse=1024, num_fine=args.num_points)
        else:
            raise NotImplementedError

        load_model(args, net, None, log_test, train=False)
        net.cuda()
        net.eval()
        log_test.log_string("Testing...")
        pcd_file = h5py.File(os.path.join(args.model_dir, '%s_pcds.h5' % epoch.split('.')[0]), 'w')
        pcd_file.create_dataset('output_pcds', (1200, args.num_points, 3))
        test_loss_cd_p = AverageValueMeter()
        test_loss_cd_t = AverageValueMeter()
        test_loss_emd = AverageValueMeter()
        test_f1_score = AverageValueMeter()
        test_loss_cat = torch.zeros([8, 4], dtype=torch.float32).cuda()
        cat_num = torch.ones([8, 1], dtype=torch.float32).cuda() * 150
        with torch.no_grad():
            for i, data in enumerate(dataloader_test):
                if args.use_mean_feature == 0:
                    label, inputs, gt = data
                    mean_feature = None
                else:
                    label, inputs, gt, mean_feature = data
                    mean_feature = mean_feature.float().cuda()

                inputs = inputs.float().cuda()
                gt = gt.float().cuda()
                inputs = inputs.transpose(2, 1).contiguous()

                if args.model_name == 'ecg':
                    _, output = net(inputs)
                else:
                    raise NotImplementedError

                # save pcd
                pcd_file['output_pcds'][args.batch_size * i:args.batch_size * (i + 1), :, :] = output.cpu().numpy()

                # EMD
                dist, _ = EMD(output, gt, 0.004, 3000)
                emd = torch.sqrt(dist).mean(1)

                # CD
                dist1, dist2, _, _ = chamLoss(gt, output)
                cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
                cd_t = dist1.mean(1) + dist2.mean(1)

                # f1
                f1, _, _ = fscore(dist1, dist2)

                test_loss_cd_p.update(cd_p.mean().item())
                test_loss_cd_t.update(cd_t.mean().item())
                test_loss_emd.update(emd.mean().item())
                test_f1_score.update(f1.mean().item())

                for j, l in enumerate(label):
                    test_loss_cat[int(l), 0] += cd_p[int(j)]
                    test_loss_cat[int(l), 1] += cd_t[int(j)]
                    test_loss_cat[int(l), 2] += emd[int(j)]
                    test_loss_cat[int(l), 3] += f1[int(j)]

                if i % 100 == 0:
                    log_test.log_string('test [%d/%d]' % (i, dataset_length / args.batch_size))

            # Per cat loss:
            for i in range(8):
                log_test.log_string('CD_p: %f, CD_t: %f, EMD: %f F1: %f' % (test_loss_cat[i, 0] / cat_num[i] * 10000,
                                                                            test_loss_cat[i, 1] / cat_num[i] * 10000,
                                                                            test_loss_cat[i, 2] / cat_num[i] * 10000,
                                                                            test_loss_cat[i, 3] / cat_num[i]))

            log_test.log_string('Overview results:')
            log_test.log_string(
                'CD_p: %f, CD_t: %f, EMD: %f F1: %f' % (test_loss_cd_p.avg, test_loss_cd_t.avg, test_loss_emd.avg,
                                                        test_f1_score.avg))
    pcd_file.close()
    log_test.close()
