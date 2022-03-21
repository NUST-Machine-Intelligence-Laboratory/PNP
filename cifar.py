import os
import sys
import pathlib
import time
import datetime
import argparse
import shutil
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torch.cuda.amp import autocast, GradScaler
from utils.core import accuracy, evaluate
from utils.builder import *
from utils.utils import *
from utils.meter import AverageMeter
from utils.logger import Logger, print_to_logfile, print_to_console
from utils.loss import *
from utils.module import MLPHead
from utils.plotter import plot_results
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import matplotlib.pyplot as plt
LOG_FREQ = 1


class CLDataTransform(object):
    def __init__(self, transform_weak, transform_strong):
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong

    def __call__(self, sample):
        x_w1 = self.transform_weak(sample)
        x_w2 = self.transform_weak(sample)
        x_s = self.transform_strong(sample)
        return x_w1, x_w2, x_s


class CNN(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25, momentum=0.1, activation='tanh'):
        self.dropout_rate = dropout_rate
        self.momentum = momentum
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=self.momentum),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=self.momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=self.momentum),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=self.momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(196, momentum=self.momentum),
            nn.ReLU(),
            nn.Conv2d(196, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, momentum=self.momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classfier_head = MLPHead(256, mlp_scale_factor=2, projection_size=n_outputs)
        self.proba_head = torch.nn.Sequential(
            MLPHead(256, mlp_scale_factor=1, projection_size=3, init_method='He', activation=activation),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        logits = self.classfier_head(x)
        prob = self.proba_head(x)
        return {'logits': logits, 'prob': prob}


def save_current_script(log_dir):
    current_script_path = __file__
    shutil.copy(current_script_path, log_dir)


def record_network_arch(result_dir, net):
    with open(f'{result_dir}/network.txt', 'w') as f:
        f.writelines(net.__repr__())


def get_smoothed_label_distribution(labels, num_class, epsilon):
    smoothed_label = torch.full(size=(labels.size(0), num_class), fill_value=epsilon / (num_class - 1))
    smoothed_label.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1).cpu(), value=1 - epsilon)
    return smoothed_label.to(labels.device)


def build_logger(params):
    logger_root = f'Results/{params.synthetic_data}'
    if not os.path.isdir(logger_root):
        os.makedirs(logger_root, exist_ok=True)
    percentile = int(params.closeset_ratio * 100)
    noise_condition = f'symm_{percentile:2d}' if params.noise_type == 'symmetric' else f'asym_{percentile:2d}'
    logtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = os.path.join(logger_root, noise_condition, params.project, f'{params.log}-{logtime}')
    logger = Logger(logging_dir=result_dir, DEBUG=True)
    logger.set_logfile(logfile_name='log.txt')
    save_config(params, f'{result_dir}/params.cfg')
    save_params(params, f'{result_dir}/params.json', json_format=True)
    save_current_script(result_dir)
    logger.msg(f'Result Path: {result_dir}')
    return logger, result_dir


def build_model_optim_scheduler(params, device, build_scheduler=True):
    assert params.dataset.startswith('cifar')
    n_classes = int(params.n_classes * (1 - params.openset_ratio))
    net = CNN(input_channel=3, n_outputs=n_classes, activation='leaky relu' if params.activation == 'l_relu' else params.activation)
    if params.opt == 'sgd':
        optimizer = build_sgd_optimizer(net.parameters(), params.lr, params.weight_decay, nesterov=True)
    elif params.opt == 'adam':
        optimizer = build_adam_optimizer(net.parameters(), params.lr)
    else:
        raise AssertionError(f'{params.opt} optimizer is not supported yet.')
    if build_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True, threshold=1e-4)
    else:
        scheduler = None
    return net.to(device), optimizer, scheduler, n_classes


def build_lr_plan(params, factor=10, decay='linear'):
    epoch_decay_start = 80
    lr_plan = [params.lr] * params.epochs
    for i in range(0, params.warmup_epochs):
        lr_plan[i] *= factor
    for i in range(epoch_decay_start, params.epochs):
        if decay == 'linear':
            lr_plan[i] = float(params.epochs - i) / (params.epochs - params.warmup_epochs) * params.lr  # linearly decay
        elif decay == 'cosine':
            lr_plan[i] = 0.5 * params.lr * (1 + math.cos((i - params.warmup_epochs + 1) * math.pi / (params.epochs - params.warmup_epochs + 1)))  # cosine decay
        else:
            raise AssertionError(f'lr decay method: {decay} is not implemented yet.')
    return lr_plan
    
    
def build_dataset_loader(params):
    assert params.dataset.startswith('cifar')
    transform = build_transform(rescale_size=params.rescale_size, crop_size=params.crop_size)
    if params.dataset == 'cifar100':
        dataset = build_cifar100n_dataset(os.path.join(params.database, params.dataset), CLDataTransform(transform['cifar_train'], transform['cifar_train_strong_aug']), transform['cifar_test'], noise_type=params.noise_type, openset_ratio=params.openset_ratio, closeset_ratio=params.closeset_ratio)
    else:
        raise AssertionError(f'{params.dataset} dataset is not supported yet.')
    train_loader = DataLoader(dataset['train'], batch_size=params.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(dataset['test'], batch_size=16, shuffle=False, num_workers=8, pin_memory=False)
    return dataset, train_loader, test_loader


def wrapup_training(result_dir, best_accuracy):
    stats = get_stats(f'{result_dir}/log.txt')
    with open(f'{result_dir}/result_stats.txt', 'w') as f:
        f.write(f"valid epochs: {stats['valid_epoch']}\n")
        if 'mean' in stats.keys():
            f.write(f"mean: {stats['mean']:.4f}, std: {stats['std']:.4f}\n")
        else:
            f.write(f"mean1: {stats['mean1']:.4f}, std2: {stats['std1']:.4f}\n")
            f.write(f"mean2: {stats['mean2']:.4f}, std2: {stats['std2']:.4f}\n")
    os.rename(result_dir, f'{result_dir}-bestAcc_{best_accuracy:.4f}')


def main(cfg, device):
    init_seeds(0)
    cfg.use_fp16 = False if device.type == 'cpu' else cfg.use_fp16

    logger, result_dir = build_logger(cfg)
    net, optimizer, scheduler, n_classes = build_model_optim_scheduler(cfg, device, build_scheduler=False)
    lr_plan = build_lr_plan(cfg, factor=cfg.warmup_lr_scale, decay=cfg.lr_decay)
    dataset, train_loader, test_loader = build_dataset_loader(cfg)

    logger.msg(f"Categories: {n_classes}, Training Samples: {dataset['n_train_samples']}, Testing Samples: {dataset['n_test_samples']}")
    logger.msg(f"Noise Type: {dataset['train'].noise_type}, Openset Noise Ratio: {dataset['train'].openset_noise_ratio}, Closedset Noise Ratio: {dataset['train'].closeset_noise_rate}")
    logger.msg(f'Optimizer: {cfg.opt}')
    record_network_arch(result_dir, net)

    if cfg.loss_func_aux == 's-mae':
        aux_loss_func = F.smooth_l1_loss
    elif cfg.loss_func_aux == 'mae':
        aux_loss_func = F.l1_loss
    elif cfg.loss_func_aux == 'mse':
        aux_loss_func = F.mse_loss
    else:
        raise AssertionError(f'{cfg.loss_func_aux} loss is not supported for auxiliary loss yet.')

    # meters -----------------------------------------------------------------------------------------------------------------------------------------
    train_loss = AverageMeter()
    train_accuracy = AverageMeter()
    epoch_train_time = AverageMeter()
    best_accuracy, best_epoch = 0.0, None
    scaler = GradScaler()
    iters_to_accumulate = round(64/cfg.batch_size) if cfg.use_grad_accumulate and cfg.batch_size < 64 else 1
    logger.msg(f'Accumulate gradients every {iters_to_accumulate} iterations --> Acutal batch size is {cfg.batch_size * iters_to_accumulate}')

    entropy_normalize_factor = entropy(torch.ones(n_classes) / n_classes).item()
    # training ---------------------------------------------------------------------------------------------------------------------------------------
    for epoch in range(0, cfg.epochs):
        start_time = time.time()

        net.train()
        adjust_lr(optimizer, lr_plan[epoch])
        optimizer.zero_grad()
        train_loss.reset()
        train_accuracy.reset()

        # train this epoch
        pbar = tqdm(train_loader, ncols=150, ascii=' >', leave=False, desc='training')
        for it, sample in enumerate(pbar):
            curr_lr = [group['lr'] for group in optimizer.param_groups][0]
            # torch.autograd.set_detect_anomaly(True)

            s = time.time()

            indices = sample['index']
            x, x_w, x_s = sample['data']
            x, x_w, x_s = x.to(device), x_w.to(device), x_s.to(device)
            y = sample['label'].to(device)

            with autocast(cfg.use_fp16):
                output = net(x)
                logits = output['logits']
                probs = logits.softmax(dim=1)
                train_acc = accuracy(logits, y, topk=(1,))

                logits_s = net(x_s)['logits']
                logits_w = net(x_w)['logits']

                type_prob = output['prob'].softmax(dim=1)  # (N, 3)
                clean_pred_prob = type_prob[:, 0]
                idn_pred_prob = type_prob[:, 1]
                ood_pred_prob = type_prob[:, 2]

                pbar.set_postfix_str(f'TrainAcc: {train_accuracy.avg:3.2f}%; TrainLoss: {train_loss.avg:3.2f}')
                given_labels = get_smoothed_label_distribution(y, n_classes, epsilon=cfg.epsilon)
                if epoch < cfg.warmup_epochs:
                    pbar.set_description(f'WARMUP TRAINING (lr={curr_lr:.3e})')
                    loss = 0.5 * cross_entropy(logits, given_labels, reduction='mean') + 0.5 * cross_entropy(logits_w, given_labels, reduction='mean')
                else:
                    pbar.set_description(f'ROBUST TRAINING (lr={curr_lr:.3e})')

                    # strong aug, "NeurIPS 2020 - Unsupervised Data Augmentation for Consistency Training"
                    probs_s = logits_s.softmax(dim=1)
                    probs_w = logits_w.softmax(dim=1)
                    with torch.no_grad():
                        mean_pred_prob_dist = (probs + probs_w + given_labels) / 3
                        sharpened_target_s = (mean_pred_prob_dist / cfg.temperature).softmax(dim=1)
                        flattened_target_s = (mean_pred_prob_dist * cfg.temperature).softmax(dim=1)

                    # classification loss
                    loss_clean = 0.5 * cross_entropy(logits, given_labels, reduction='none') + 0.5 * cross_entropy(logits_w, given_labels, reduction='none')
                    loss_idn = cross_entropy(logits_s, sharpened_target_s, reduction='none') * cfg.rho
                    loss_ood = cross_entropy(logits_s, flattened_target_s, reduction='none') * cfg.beta

                    # entropy loss
                    loss_entropy = 0.5 * entropy_loss(logits, reduction='none') + 0.5 * entropy_loss(logits_w, reduction='none')
                    loss_clean += loss_entropy * cfg.alpha

                    # consistency loss
                    loss_cons = symmetric_kl_div(probs, probs_w)

                    type_target = torch.nn.functional.one_hot(type_prob.max(dim=1)[1], 3)
                    if_clean = type_target[:, 0]
                    if_idn = type_target[:, 1]
                    if_ood = type_target[:, 2]
                    if cfg.weighting == 'soft':
                        # soft seletcion / weighting
                        loss_cls = loss_clean * clean_pred_prob + loss_idn + idn_pred_prob + loss_ood * ood_pred_prob
                        if cfg.neg_cons:
                            loss_cons = loss_cons * (clean_pred_prob + idn_pred_prob - ood_pred_prob)
                        else:
                            loss_cons = loss_cons * (clean_pred_prob + idn_pred_prob)
                        loss_cons = loss_cons.mean()
                    else:
                        # hard seletcion / weighting
                        loss_cls = loss_clean * if_clean + loss_idn + if_idn + loss_ood * if_ood
                        if cfg.neg_cons:
                            loss_cons = loss_cons * if_clean + loss_cons * if_idn - loss_cons * if_ood
                            loss_cons = loss_cons.mean()
                        else:
                            loss_cons = loss_cons * if_clean + loss_cons * if_idn
                            n_clean, n_idn = torch.nonzero(if_clean, as_tuple=False).shape[0], torch.nonzero(if_idn, as_tuple=False).shape[0]
                            loss_cons = loss_cons.sum() / (n_clean + n_idn) if n_clean + n_idn > 0 else 0
                    loss_cls = loss_cls.mean()
                    
                    # auxiliary loss
                    with torch.no_grad():
                        clean_probs = (1 - js_div(probs, given_labels))
                        ood_probs = js_div(probs, probs_w) * cfg.eta + entropy(probs) / entropy_normalize_factor * (1 - cfg.eta)
                    loss_aux_clean = aux_loss_func(clean_pred_prob, clean_probs)
                    loss_aux_ood = aux_loss_func(ood_pred_prob, ood_probs)
                    loss_aux = loss_aux_clean + loss_aux_ood

                    loss = loss_cls + cfg.gamma * loss_aux + cfg.omega * loss_cons

                    # debug
                    # loss_content = f'loss: {loss.item():3.2f}; loss_cls: (clean={loss_clean.mean():3.2f}, idn={loss_idn.mean():3.2f}, ood={loss_ood.mean():3.5f}); loss_aux: (clean={loss_aux_clean:3.2f}, ood={loss_aux_ood:3.2f})'
                    # with open(f'{result_dir}/debug-loss.txt', 'a') as f:
                    #     f.write(f'{loss_content}\n')
                    # prob_content = f'prob_clean: {clean_pred_prob.mean().item():3.5f}({clean_probs.mean().item():3.5f}); prob_id: {idn_pred_prob.mean().item():3.5f}; prob_ood: {ood_pred_prob.mean().item():3.5f}({ood_probs.mean().item():3.5f})'
                    # with open(f'{result_dir}/debug-prob.txt', 'a') as f:
                    #     f.write(f'{prob_content}\n')

            # if iters_to_accumulate > 0:
            #     loss = loss / iters_to_accumulate  # ???

            scaler.scale(loss).backward()

            if (it + 1) % iters_to_accumulate == 0:
                try:
                    scaler.step(optimizer)
                except RuntimeError:  # in case of "RuntimeError: Function 'CudnnBatchNormBackward' returned nan values in its 0th output."
                    logger.msg('Runtime Error occured! Have unscaled losses and clipped grads before optimizing!')
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2, norm_type=2.0)
                    scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_accuracy.update(train_acc[0], x.size(0))
            train_loss.update(loss.item(), x.size(0))
            epoch_train_time.update(time.time() - s, 1)
            if (cfg.log_freq is not None and (it + 1) % cfg.log_freq == 0) or (it + 1 == len(train_loader)):
                total_mem = torch.cuda.get_device_properties(0).total_memory / 2**30
                mem = torch.cuda.memory_reserved() / 2**30
                console_content = f"Epoch:[{epoch + 1:>3d}/{cfg.epochs:>3d}]  " \
                                f"Iter:[{it + 1:>4d}/{len(train_loader):>4d}]  " \
                                f"Train Accuracy:[{train_accuracy.avg:6.2f}]  " \
                                f"Loss:[{train_loss.avg:4.4f}]  " \
                                f"GPU-MEM:[{mem:6.3f}/{total_mem:6.3f} Gb]  " \
                                f"{epoch_train_time.avg:6.2f} sec/iter"
                logger.debug(console_content)

        # evaluate this epoch
        eval_result = evaluate(test_loader, net, device)
        test_accuracy = eval_result['accuracy']
        test_loss = eval_result['loss']
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch + 1
            if cfg.save_model:
                torch.save(net.state_dict(), f'{result_dir}/best_epoch.pth')
                torch.save(net, f'{result_dir}/best_model.pth')

        # logging this epoch
        runtime = time.time() - start_time
        logger.info(f'epoch: {epoch + 1:>3d} | '
                    f'train loss: {train_loss.avg:>6.4f} | '
                    f'train accuracy: {train_accuracy.avg:>6.3f} | '
                    f'test loss: {test_loss:>6.4f} | '
                    f'test accuracy: {test_accuracy:>6.3f} | '
                    f'epoch runtime: {runtime:6.2f} sec | '
                    f'best accuracy: {best_accuracy:6.3f} @ epoch: {best_epoch:03d}')
        plot_results(result_file=f'{result_dir}/log.txt', layout='2x2')

    wrapup_training(result_dir, best_accuracy)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--synthetic-data', type=str, default='cifar80no')
    parser.add_argument('--noise-type', type=str, default='symmetric')
    parser.add_argument('--closeset-ratio', type=float, default='0.8')
    parser.add_argument('--gpu', type=str, required=True)
    parser.add_argument('--net', type=str, default='cnn')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr-decay', type=str, default='cosine')
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--warmup-lr-scale', type=float, default=10.0)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--use-fp16', action='store_true')
    parser.add_argument('--use-grad-accumulate', action='store_true')
    
    parser.add_argument('--project', type=str, default='')
    parser.add_argument('--log', type=str, default='PENIOC')
    parser.add_argument('--epsilon', type=float, default=0.5)

    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--eta', type=float, default=0.5, help='hyper-parameter for balancing target ood probability')
    parser.add_argument('--alpha', type=float, default=0.0, help='weight for the entropy loss (clean)')
    parser.add_argument('--beta', type=float, default=1.0, help='weight for the ood classification loss')
    parser.add_argument('--gamma', type=float, default=1.0, help='weight for the auxiliary loss')
    parser.add_argument('--omega', type=float, default=0.1, help='weight for the consistency loss')

    parser.add_argument('--rho', type=float, default=1.0, help='weight for the idn loss')  # TODO: delete


    parser.add_argument('--loss-func-aux', type=str, default='mae')
    parser.add_argument('--weighting', type=str, default='soft')
    parser.add_argument('--neg-cons', action='store_true')

    parser.add_argument('--activation', type=str, default='tanh')
    parser.add_argument('--ablation', action='store_true')
    
    args = parser.parse_args()

    config = load_from_cfg(args.config)
    override_config_items = [k for k, v in args.__dict__.items() if k != 'config' and v is not None]
    for item in override_config_items:
        config.set_item(item, args.__dict__[item])
    
    assert config.temperature <= 1 and config.temperature > 0, f'temperture for sharpening operation should be in (0, 1], but the currect value is {config.temperature}.'
    assert config.synthetic_data in ['cifar100nc', 'cifar80no']
    assert config.noise_type in ['symmetric', 'asymmetric']
    config.openset_ratio = 0.0 if config.synthetic_data == 'cifar100nc' else 0.2
    if config.ablation:
        config.project = f'ablation/{config.project}'
    config.log_freq = LOG_FREQ
    print(config)
    return config


if __name__ == '__main__':
    params = parse_args()
    dev = set_device(params.gpu)
    script_start_time = time.time()
    main(params, dev)
    script_runtime = time.time() - script_start_time
    print(f'Runtime of this script {str(pathlib.Path(__file__))} : {script_runtime:.1f} seconds ({script_runtime/3600:.3f} hours)')
