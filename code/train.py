import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from model.model import Net
from config import cfg, merge_cfg_from_file
from LRScheduler.LRScheduler import BuildLRScheduler
from data.DataLoader import MakeTrainLoader


def TrainEpoch(epoch_idx, train_loader, model, loss_funcs, optimizer, lr_scheduler, device):
    model.train()

    for batch_idx, input_data in enumerate(train_loader):
        inputs, targets = input_data[0].to(device), input_data[1].to(device)
        logits = model(inputs)

        loss_sum = 0
        for idx in range(len(logits)):
            loss = loss_funcs(logits, targets)
            loss_sum += loss

        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()
        lr_scheduler.step(epoch=epoch_idx - 1, batch=batch_idx + 1)


def TrainTest(cfg):
    os.environ['CUDA_DEVICES_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(item) for item in cfg.TASK.GPUS])
    device = torch.device('cuda' if len(cfg.TASK.GPUS) > 0 else 'cpu')

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
    param_groups = [{'params': params, 'lr': 0.001}]
    optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=5e-4)
    # lr scheduler
    lr_scheduler = BuildLRScheduler(optimizer, [5, 10], 0.1)

    # start train
    for epoch_idx in range(0, 20):
        TrainEpoch(epoch_idx, train_loader,model, loss_funcs, optimizer, lr_scheduler, device)


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

    TrainTest(cfg)

