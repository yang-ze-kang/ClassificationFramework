from .datasets.dataset import *

cudas = '3'
ngpus = 1

weight_save_interval = 10


model = dict(
    model_name='vit_base_patch16_224',
    num_classes=2,
    pretrained = True,
)

optimizer = dict(
    epochs = 100,
    resume = None,
    weight_decay = 5e-4,
    max_learning_rate = 1e-4,
    consine_T = 100,
)


