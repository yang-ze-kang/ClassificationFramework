from torch.utils.data import DataLoader, WeightedRandomSampler
from utils.datasets import DataGenerator
from utils.log import Log
from models import create_model
import torch.nn.functional as F
import torch.nn as nn
import argparse
import torch
from tqdm import tqdm
import os


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(model, dataloader, cfg):
    log = Log(cfg.config_name, cfg.model_name)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                                 cfg.optimizer['max_learning_rate'],
                                 weight_decay=cfg.optimizer['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.optimizer['consine_T'], eta_min=0)
    min_loss = 1e8
    if cfg.optimizer['resume'] is None:
        start = 1
    epochs = cfg.optimizer['epochs']+1
    for epoch in range(start, epochs):
        train_loss = 0
        val_loss = 0
        print('====Start Train====')
        model.train()
        num_t = 0
        num_sum = 0
        with tqdm(total=cfg.dataset['train']['num']/cfg.dataset['batch_size'],
                  desc=f'Epoch {epoch}/{epochs}',
                  mininterval=0.3) as pbar:
            for iter, batch in enumerate(dataloader['train']):
                iter += 1
                images, labels = batch
                images = images.to(cfg.device)
                labels = labels.to(cfg.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_function(outputs, labels)
                loss_value = loss.item()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
                    num_t += torch.sum(labels == preds).item()
                    num_sum += len(labels)
                train_loss += loss_value
                if iter % 100 == 0:
                    pbar.set_postfix(
                        **{
                            'loss': loss_value/cfg.dataset['batch_size'],
                            'accuracy': num_t/num_sum,
                            'lr': get_lr(optimizer)
                        })
                    pbar.update(100)
        train_accuracy = num_t/num_sum
        train_loss = train_loss/iter
        print('====End Train====')
        print('====Start Validate====')
        model.eval()
        num_t = 0
        num_sum = 0
        with tqdm(total=cfg.dataset['val']['num']/cfg.dataset['batch_size'],
                  desc=f'Epoch {epoch}/{epochs}',
                  mininterval=0.3) as pbar:
            for iter, batch in enumerate(dataloader['val']):
                iter += 1
                images, labels = batch
                images = images.to(cfg.device)
                labels = labels.to(cfg.device)
                outputs = model(images)
                loss = loss_function(outputs, labels)
                loss_value = loss.item()
                val_loss += loss_value
                preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
                num_t += torch.sum(labels == preds).item()
                num_sum += len(labels)
                if iter % 100 == 0:
                    pbar.set_postfix(
                        **{
                            'loss': loss_value/cfg.dataset['batch_size'],
                            'accuracy': num_t/num_sum
                        })
                    pbar.update(100)
        val_accuracy = num_t/num_sum
        val_loss = val_loss/iter
        if min_loss > val_loss:
            min_loss = val_loss
            log.save_model(model)
        if epoch % cfg.weight_save_interval == 0:
            log.save_model(model, str(epoch))
        else:
            log.save_model(model, 'last')
        print('====End Validate====')
        print('Epoch:' + str(epoch) + '/' + str(epochs))
        print('Total Loss: %.4f || Val Loss: %.4f || Total accuracy %.4f || Val accuracy %.4f' %
              (train_loss, val_loss, train_accuracy, val_accuracy))
        lr_scheduler.step()
        log.append_loss(train_loss, val_loss)
        log.append_accuracy(train_accuracy, val_accuracy)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', default='configs/config',
                        help='train config file path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    cfg = args.config
    cfgs = cfg.split('/')
    assert len(cfgs) == 2
    cfg = getattr(__import__(cfgs[0]), cfgs[1])
    cfg.config_name = cfgs[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.cudas
    cfg.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    if isinstance(cfg.model, dict):
        cfg.model_name = cfg.model['model_name']
        model = create_model(cfg.model)
    else:
        model = cfg.model
    model.to(cfg.device)
    if cfg.ngpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(cfg.ngpus)))
    train_dataset = DataGenerator(
        cfg.dataset['train']['file_path'], cfg.dataset['train']['data_transform'], type='train')
    val_dataset = DataGenerator(
        cfg.dataset['val']['file_path'], cfg.dataset['val']['data_transform'], type='val')
    cfg.dataset['train']['num'] = len(train_dataset)
    cfg.dataset['val']['num'] = len(val_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.dataset['batch_size'],
                              shuffle=cfg.dataset['train']['shuffle'],
                              pin_memory=True,
                              sampler=None,
                              num_workers=cfg.dataset['num_works'],
                              collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.dataset['batch_size'],
                            shuffle=cfg.dataset['train']['shuffle'],
                            pin_memory=True,
                            sampler=None,
                            num_workers=cfg.dataset['num_works'],
                            collate_fn=val_dataset.collate_fn)
    dataloader = {'train': train_loader, 'val': val_loader}
    train(model, dataloader, cfg)
