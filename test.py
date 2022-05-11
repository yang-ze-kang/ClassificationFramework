from utils.datasets import DataGenerator
from models import create_model
import torch.nn.functional as F
import torch.nn as nn
import argparse
import torch
from torch.utils.data import DataLoader
from utils.metric import MutiClassMetric
from tqdm import tqdm
import os


def train(model, dataloader, cfg):
    print('====Start Validate====')
    model.eval()
    metric = MutiClassMetric(num_classes=2, labels_name=['dry', 'wet'])
    num_t = 0
    num_sum = 0
    with tqdm(total=cfg.dataset['test']['num']/cfg.dataset['batch_size'],
              mininterval=0.3) as pbar:
        for iter, batch in enumerate(dataloader):
            iter += 1
            images, labels = batch
            images = images.to(cfg.device)
            labels = labels.to('cpu').numpy()
            outputs = model(images)
            scores = F.softmax(outputs, dim=1).to('cpu').detach().numpy()
            print(scores.shape)
            print(labels.shape)
            metric.update(scores, labels)
            if iter % 50 == 0:
                pbar.update(50)
    # metric.summary()
    print(metric.get_auc())
    metric.plot_confusion_matrix()
    metric.plot_roc()


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', default='configs/config',
                        help='train config file path')
    parser.add_argument('--eval_weights', default='/home/yzk/ClassificationFramework/logs/config_2022-05-11 20:42/model/best.pth',
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
    model.load_state_dict(torch.load(
        args.eval_weights, map_location=cfg.device), strict=False)
    if cfg.ngpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(cfg.ngpus)))
    dataset = DataGenerator(
        cfg.dataset['test']['file_path'], cfg.dataset['test']['data_transform'], type='test')
    cfg.dataset['test']['num'] = len(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=cfg.dataset['batch_size'],
                            shuffle=cfg.dataset['test']['shuffle'],
                            pin_memory=True,
                            sampler=None,
                            num_workers=cfg.dataset['num_works'],
                            collate_fn=dataset.collate_fn)
    train(model, dataloader, cfg)
