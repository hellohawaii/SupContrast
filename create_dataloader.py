from torchvision import transforms, datasets
import torch
from config import get_contrast_transform_config, get_linear_train_transform_config, get_linear_test_transform_config
from torch.utils.data import WeightedRandomSampler
import logging
from torch.utils.data.sampler import Sampler

logger = logging.getLogger(__name__)

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def build_transforms(data_config):  # modify from MICCAI
    transform_list = [transforms.Resize(size=data_config.img_size, interpolation=transforms.InterpolationMode.BILINEAR)]
    if 'rotation' in data_config:
        transform_list += [
            transforms.RandomRotation(degrees=(-180, 180),
                                      interpolation=transforms.InterpolationMode.BILINEAR, expand=True),
            # crop again because of the rotation
            transforms.CenterCrop(size=data_config.img_size)]

    if 'flip' in data_config:
        transform_list += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    if 'heavy_data_augmentation' in data_config:
        data_heavy_aug_config = data_config['heavy_data_augmentation']
        data_augmentation_transform = []
        if 'color_jitter' in data_heavy_aug_config:
            data_augmentation_transform.append(transforms.ColorJitter(
                brightness=data_heavy_aug_config['color_jitter']['brightness'],
                contrast=data_heavy_aug_config['color_jitter']['contrast'],
                saturation=data_heavy_aug_config['color_jitter']['saturation'],
                hue=data_heavy_aug_config['color_jitter']['hue']
            ))
        if 'resized_crop' in data_heavy_aug_config:
            data_augmentation_transform.append(transforms.RandomResizedCrop(
                size=(data_heavy_aug_config['resized_crop']['input_size'], data_heavy_aug_config['resized_crop']['input_size']),
                scale=data_heavy_aug_config['resized_crop']['scale'],
                ratio=data_heavy_aug_config['resized_crop']['ratio']
            ))
        if 'affine' in data_heavy_aug_config:
            data_augmentation_transform.append(transforms.RandomAffine(
                degrees=data_heavy_aug_config['affine']['degrees'],
                translate=data_heavy_aug_config['affine']['translate']
            ))
        if 'gray' in data_heavy_aug_config:
            data_augmentation_transform.append(transforms.RandomGrayscale(data_heavy_aug_config['gray']))
        transform_list += data_augmentation_transform

    transform_list.append(transforms.ToTensor())
    if 'mean' in data_config and 'std' in data_config:
        mean, std = data_config['mean'], data_config['std']
        transform_list.append(transforms.Normalize(mean, std))
    transform = transforms.Compose(transform_list)

    return transform


def set_contrast_loader(opt):
    # construct data loader
    contrast_transform_config = get_contrast_transform_config()
    logger.info("********************CONTRAST TRAIN TRANSFORM***********************")
    logger.info(contrast_transform_config)
    train_transform = build_transforms(contrast_transform_config)

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                            transform=TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader


# copy from MICCAI pytorch-classification
class ScheduledWeightedSampler(Sampler):
    def __init__(self, dataset, decay_rate):
        self.dataset = dataset
        self.decay_rate = decay_rate

        self.num_samples = len(dataset)
        self.targets = [sample[1] for sample in dataset.imgs]
        self.class_weights = self.cal_class_weights()

        self.epoch = 0
        self.w0 = torch.as_tensor(self.class_weights, dtype=torch.double)
        self.wf = torch.as_tensor([1] * len(self.dataset.classes), dtype=torch.double)
        self.sample_weight = torch.zeros(self.num_samples, dtype=torch.double)
        for i, _class in enumerate(self.targets):
            self.sample_weight[i] = self.w0[_class]

    def step(self):
        if self.decay_rate < 1:
            self.epoch += 1
            factor = self.decay_rate**(self.epoch - 1)
            self.weights = factor * self.w0 + (1 - factor) * self.wf
            logger.info("current_weights is %s" % self.weights)
            for i, _class in enumerate(self.targets):
                self.sample_weight[i] = self.weights[_class]

    def __iter__(self):
        return iter(torch.multinomial(self.sample_weight, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples

    def cal_class_weights(self):
        num_classes = len(self.dataset.classes)
        classes_idx = list(range(num_classes))
        class_count = [self.targets.count(i) for i in classes_idx]
        weights = [self.num_samples / class_count[i] for i in classes_idx]
        min_weight = min(weights)
        class_weights = [weights[i] / min_weight for i in classes_idx]
        return class_weights


def set_linear_loader(opt):
    # construct data loader
    linear_train_transform_config = get_linear_train_transform_config()
    logger.info("********************LINEAR TRAIN TRANSFORM***********************")
    logger.info(linear_train_transform_config)
    linear_test_transform_config = get_linear_test_transform_config()
    logger.info("********************LINEAR TEST TRANSFORM***********************")
    logger.info(linear_test_transform_config)
    train_transform = build_transforms(linear_train_transform_config)
    val_transform = build_transforms(linear_test_transform_config)

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False,
                                        transform=val_transform)
    elif opt.dataset == 'path':
        # train_dataset = datasets.ImageFolder(root=opt.data_folder + '/train',
        #                                      transform=train_transform)
        # val_dataset = datasets.ImageFolder(root=opt.data_folder + '/test',
        #                                    transform=val_transform)
        train_dataset = datasets.ImageFolder(root=opt.data_folder + '/train/image/',
                                             transform=train_transform)
        val_dataset = datasets.ImageFolder(root=opt.data_folder + '/val/image',
                                           transform=val_transform)
    else:
        raise ValueError(opt.dataset)

    if opt.oversample:
        train_sampler = ScheduledWeightedSampler(train_dataset, decay_rate=opt.resample_decay_rate)
        # if opt.dacey_rate == 1 ,this is oversampling
    else:
        train_sampler = None
    # counting for num of targets
    # from collections import Counter
    # class_count = Counter(train_dataset.targets)  # use dir(train_dataset) to show() all attributes
    # class_weights = {k: 1/v for k, v in class_count.items()}
    # item_weights = [class_weights[i] for i in train_dataset.targets]
    #
    # train_sampler = WeightedRandomSampler(item_weights, len(train_dataset))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
        # num_workers=opt.num_workers, pin_memory=True, sampler=weighted_train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader, train_sampler