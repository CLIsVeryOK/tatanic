import os
import sys
import shutil
import torch
import torch.nn as nn
import torch.optim as optim

from model.model import Net
from config import cfg, merge_cfg_from_file
from LRScheduler.LRScheduler import BuildLRScheduler
from data.DataLoader import MakeTrainLoader

from log.logger import setup_logger, AverageMeter


def TrainEpoch(epoch_idx, train_loader, model, loss_funcs, optimizer, lr_scheduler, device, logger):
    model.train()

    for batch_idx, input_data in enumerate(train_loader):
        inputs, targets = input_data[0].to(device), input_data[1].to(device)
        logits = model(inputs)

        loss = loss_funcs(logits, targets)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
        logger.info('epoch: %d, loss: %f, lr: %f' % (epoch_idx, loss, lr_scheduler.get_lr()[0]))


def Valid(val_loader, model, loss_funcs, device, logger):
    model.eval()
    correct = 0
    num_samples = len(val_loader.dataset)
    with torch.no_grad():
        for batch_idx, input_data in enumerate(val_loader):
            inputs, targets = input_data[0].to(device), input_data[1].to(device)
            logits = model(inputs)

            loss = loss_funcs(logits, targets)

    precision = correct * 1. / num_samples
    logger.info('precision: %f' % precision)
    return precision


def Train(cfg):
    os.environ['CUDA_DEVICES_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(item) for item in cfg.TASK.GPUS])
    device = torch.device('cuda' if len(cfg.TASK.GPUS) > 0 else 'cpu')

    # init logger
    output_dir = os.path.join(cfg.TASK.OUTPUT_ROOT_DIR, cfg.TASK.NAME)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = setup_logger(cfg.TASK.NAME, output_dir, distributed_rank=0)

    # data loader
    train_loader = MakeTrainLoader(os.path.join(cfg.DATA.ROOT_DIR, cfg.DATA.TRAIN_PATH), cfg.DATA.TRAIN_BATCH)
    num_images = len(train_loader.dataset)
    print('total train data: ', num_images)

    # model
    model = Net().to(device)
    # loss
    loss_funcs = nn.CrossEntropyLoss().to(device)
    # optimizer
    params = [p for n, p in model.named_parameters()]
    param_groups = [{'params': params, 'lr': 0.1}]
    # optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(param_groups)
    # lr scheduler
    lr_scheduler = BuildLRScheduler(optimizer, [50, 100, 200], 0.1)

    # start train
    for epoch_idx in range(0, cfg.SOLVER.EPOCHS):
        logger.info('train epoch: {0}'.format(epoch_idx))
        TrainEpoch(epoch_idx, train_loader, model, loss_funcs, optimizer, lr_scheduler, device, logger)
        # if not epoch_idx % 10:
        #     val_loader = MakeTrainLoader(os.path.join(cfg.DATA.ROOT_DIR, cfg.DATA.VAL_PATH), cfg.DATA.VAL_BATCH)
        #     Valid(val_loader, model, loss_funcs, device, logger)


def parse_args(args):
    import argparse
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument(
        '--yaml_filepath',
        type=str,
        help='config filepath',
        default='ConfigYaml/config.yaml')

    args = parser.parse_args(args)
    return args


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    if os.path.exists(args.yaml_filepath):
        merge_cfg_from_file(args.yaml_filepath, cfg)
    else:
        raise Exception('invalid config filepath: ', args.yaml_filepath)

    Train(cfg)

