import os
import random
import torch.nn as nn
import datetime
import visdom
from data.dataset import *


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv2d') == -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1 and classname.find('BatchNorm2d') == -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def weights_init_cascade(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def vis_curve_plot(vis, log_env, train_curve, val_curves):
    val_curve_emd = val_curves['val_curve_emd']
    val_curve_cd_p = val_curves['val_curve_cd_p']
    val_curve_cd_t = val_curves['val_curve_cd_t']
    vis.line(X=np.column_stack((np.arange(len(train_curve)), np.arange(len(val_curve_emd)),
                                np.arange(len(val_curve_cd_p)), np.arange(len(val_curve_cd_t)))),
             Y=np.column_stack((np.array(train_curve), np.array(val_curve_emd), np.array(val_curve_cd_p),
                                np.array(val_curve_cd_t))),
             win='loss', opts=dict(title="emd", legend=["train_curve" + log_env, "val_curve_emd" + log_env,
                                                        "val_curve_cd_p" + log_env, "val_curve_cd_t" + log_env],
                                   markersize=2))


def epoch_log(args, log_model, train_loss_meter, val_loss_meters, best_val_losses, best_val_epochs):
    train_loss_name = 'train_loss_' + args.loss
    string1 = "%s: %f " % (train_loss_name, train_loss_meter.avg)
    string2 = "val_loss_emd: %f " % val_loss_meters["val_loss_emd"].avg
    string3 = "val_loss_cd_p: %f " % val_loss_meters["val_loss_cd_p"].avg
    string4 = "val_loss_cd_t: %f " % val_loss_meters["val_loss_cd_t"].avg
    string5 = "best_val_emd: %f in epoch %d " % (best_val_losses["best_emd_loss"], best_val_epochs["best_emd_epoch"])
    string6 = "best_val_cd_p: %f in epoch %d " % (best_val_losses["best_cd_p_loss"], best_val_epochs["best_cd_p_epoch"])
    string7 = "best_val_cd_t: %f in epoch %d " % (best_val_losses["best_cd_t_loss"], best_val_epochs["best_cd_t_epoch"])
    log_model.log_string(string1 + string2 + string3 + string4 + string5 + string6 + string7, stdout=False)


def log_setup(args):
    # log_path:
    now = datetime.datetime.now()
    subfolder_path = args.log_env + '_' + now.isoformat()[:19]
    log_root = os.path.join('log/')
    dir_name = os.path.join(log_root, '%s_%s_train' % (args.model_name, args.loss), subfolder_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # visdom log (train curve, val curve)
    vis = visdom.Visdom(log_to_filename=os.path.join(dir_name, 'visdom.log'), env=args.log_env, offline=True)
    log_model = LogString(open(os.path.join(dir_name, 'log_model.txt'), 'w'))
    log_train = LogString(open(os.path.join(dir_name, 'log_train.txt'), 'w'))
    log_train.log_string(str(args))
    return vis, log_model, log_train, dir_name


def data_setup(args, log_train):
    dataset = ShapeNetH5(train=True, npoints=args.num_points, use_mean_feature=args.use_mean_feature)
    dataset_test = ShapeNetH5(train=False, npoints=args.num_points, use_mean_feature=args.use_mean_feature)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=int(args.workers))
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=int(args.workers))
    len_dataset = len(dataset)
    len_val_dataset = len(dataset_test)
    log_train.log_string("Train Set Size: %d" % len_dataset)
    log_train.log_string("Test Set Size: %d" % len_val_dataset)
    return dataset, dataset_test, dataloader, dataloader_test


def seed_setup(args, log_train):
    # Seed
    if args.manual_seed == '':
        seed = random.randint(1, 10000)
    else:
        seed = int(args.manual_seed)
    log_train.log_string("Random Seed: %d" % seed)
    random.seed(seed)
    torch.manual_seed(seed)


def vis_curve_setup():
    train_curve = []
    val_curves = {'val_curve_emd': [],
                  'val_curve_cd_p': [],
                  'val_curve_cd_t': []}
    return train_curve, val_curves


def best_loss_setup():
    best_losses = {'best_emd_loss': 10,
                   'best_cd_p_loss': 10,
                   'best_cd_t_loss': 10}
    best_epochs = {'best_emd_epoch': 0,
                   'best_cd_p_epoch': 0,
                   'best_cd_t_epoch': 0}
    return best_losses, best_epochs


def loss_average_meter_setup():
    train_loss_meter = AverageValueMeter()
    val_loss_meters = {'val_loss_emd': AverageValueMeter(),
                       'val_loss_cd_p': AverageValueMeter(),
                       'val_loss_cd_t': AverageValueMeter()}
    return train_loss_meter, val_loss_meters


def load_model(args, net, optimizer, log_train, net_d=None, optimizer_d=None, train=True):
    if args.load_model != '':
        ckpt_path = args.load_model
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path)
            if train:
                net.module.model.load_state_dict(ckpt['net_state_dict'])
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                if net_d is not None:
                    net_d.module.load_state_dict(ckpt['D_state_dict'])
                    optimizer_d.load_state_dict(ckpt['optimizer_D_state_dict'])
            else:
                net.load_state_dict(ckpt['net_state_dict'])
            log_train.log_string("%s's weight loaded." % args.model_name)
        else:
            raise FileNotFoundError


def save_model(path, net, optimizer, net_d=None, optimizer_d=None):
    if net_d is not None:
        torch.save({'net_state_dict': net.module.model.state_dict(),
                    'D_state_dict': net_d.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizer_D_state_dict': optimizer_d.state_dict()}, path)
    else:
        torch.save({'net_state_dict': net.module.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, path)


def val(args, net, epoch, dataloader_test, log_train, best_val_losses, best_val_epochs, val_losses):
    log_train.log_string("Testing...")
    val_loss_emd = val_losses["val_loss_emd"]
    val_loss_cd_p = val_losses["val_loss_cd_p"]
    val_loss_cd_t = val_losses["val_loss_cd_t"]
    val_loss_emd.reset()
    val_loss_cd_p.reset()
    val_loss_cd_t.reset()
    net.module.model.eval()

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
            if args.model_name == 'cascade':
                _, _, _, emd2, _, cd_p2, _, cd_t2 = net(inputs, gt.contiguous(), mean_feature, 0.004, 3000)
            elif args.model_name == 'pcn':
                _, _, _, emd2, _, cd_p2, _, cd_t2 = net(inputs, gt.contiguous(), 0.004, 3000)
            elif args.model_name == 'topnet':
                _, emd2, cd_p2, cd_t2 = net(inputs, gt.contiguous(), 0.004, 3000)
            elif args.model_name == 'msn':
                _, _, _, emd2, _, cd_p2, _, cd_t2, _ = net(inputs, gt.contiguous(), 0.004, 3000)
            elif args.model_name == 'ecg':
                _, _, _, emd2, _, cd_p2, _, cd_t2, _, _ = net(inputs, gt.contiguous(), 0.004, 3000)

            val_loss_cd_p.update(cd_p2.mean().item())
            val_loss_cd_t.update(cd_t2.mean().item())
            val_loss_emd.update(emd2.mean().item())

        if val_loss_cd_p.avg < best_val_losses["best_cd_p_loss"]:
            best_val_losses["best_cd_p_loss"] = val_loss_cd_p.avg
            best_val_epochs["best_cd_p_epoch"] = epoch

        if val_loss_cd_t.avg < best_val_losses["best_cd_t_loss"]:
            best_val_losses["best_cd_t_loss"] = val_loss_cd_t.avg
            best_val_epochs["best_cd_t_epoch"] = epoch

        if val_loss_emd.avg < best_val_losses["best_emd_loss"]:
            best_val_losses["best_emd_loss"] = val_loss_emd.avg
            best_val_epochs["best_emd_epoch"] = epoch

        log_train.log_string(
            'best_cd_p_loss: %f, best_cd_t_loss: %f, best_emd_loss: %f, cur_cd_p_loss: %f cur_cd_t_loss: %f, cur_emd_loss: %f' % (
            best_val_losses["best_cd_p_loss"],
            best_val_losses["best_cd_t_loss"],
            best_val_losses["best_emd_loss"],
            val_loss_cd_p.avg,
            val_loss_cd_t.avg, val_loss_emd.avg))

        return best_val_losses, best_val_epochs


class AverageValueMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LogString(object):
    def __init__(self, file_stream):
        self.file_stream = file_stream

    def log_string(self, out_str, stdout=True):
        self.file_stream.write(out_str + '\n')
        self.file_stream.flush()
        if stdout:
            print(out_str)

    def close(self):
        self.file_stream.close()
