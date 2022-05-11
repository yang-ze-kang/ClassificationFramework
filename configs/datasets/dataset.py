from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize, ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize, ]),
    "test": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize, ]),

}

dataset = dict(
    batch_size=16,
    num_works=4,
    train=dict(
        file_path='/home/yzk/ClassificationFramework/annos/train.txt',
        num=0,
        data_transform=data_transforms['train'],
        sampler=None,
        shuffle=True,
    ),
    val=dict(
        file_path='/home/yzk/ClassificationFramework/annos/val.txt',
        num=0,
        data_transform=data_transforms['val'],
        sampler=None,
        shuffle=True,
    ),
    test=dict(
        file_path='/home/yzk/ClassificationFramework/annos/test.txt',
        num=0,
        data_transform=data_transforms['test'],
        sampler=None,
        shuffle=True,
    ),
)
