import torch
from torchvision import transforms
from torchvision.transforms import v2


def data_augmentation(data_augmentation_mode=0, edge_size=384):
    if data_augmentation_mode == -1:
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation((0, 180)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.CenterCrop(700),  # center area for classification
                transforms.Resize([edge_size, edge_size]),
                transforms.ColorJitter(brightness=0.15, contrast=0.3, saturation=0.3, hue=0.06),
                # HSL shift operation
                transforms.ToTensor()
            ]),
            'val': transforms.Compose([
                transforms.CenterCrop(700),
                transforms.Resize([edge_size, edge_size]),
                transforms.ToTensor()
            ]),
        }
    
    elif data_augmentation_mode == 0:
        # Fast and modern version
        # antialias will be set to True for PIL image on default in torchvision 0.17
        data_transforms = {
            'train': v2.Compose([
                v2.RandomRotation((0, 180)),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.CenterCrop(700),
                v2.Resize([edge_size, edge_size], antialias=True),
                v2.ColorJitter(brightness=0.15, contrast=0.3, saturation=0.3, hue=0.06),
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True)
            ]),
            'val': v2.Compose([
                v2.CenterCrop(700),
                v2.Resize([edge_size, edge_size], antialias=True),
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True)
            ])
        }
        
    elif data_augmentation_mode == 1:  # for future usage
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize([edge_size, edge_size]),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.3, saturation=0.3, hue=0.06),
                # HSL shift operation
                transforms.ToTensor()
            ]),
            'val': transforms.Compose([
                transforms.Resize([edge_size, edge_size]),
                transforms.ToTensor()
            ]),
        }

    elif data_augmentation_mode == 2:  # for future usage
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation((0, 180)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.CenterCrop(360),  # center area for classification
                transforms.Resize([edge_size, edge_size]),
                transforms.ColorJitter(brightness=0.15, contrast=0.3, saturation=0.3, hue=0.06),
                # HSL shift operation
                transforms.ToTensor()
            ]),
            'val': transforms.Compose([
                transforms.CenterCrop(360),
                transforms.Resize([edge_size, edge_size]),
                transforms.ToTensor()
            ]),
        }

    elif data_augmentation_mode == 3:
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Resize([edge_size, edge_size]),
                transforms.ColorJitter(brightness=0.15, contrast=0.3, saturation=0.3, hue=0.06),
                # HSL shift operation
                transforms.ToTensor()
            ]),
            'val': transforms.Compose([
                transforms.Resize([edge_size, edge_size]),
                transforms.ToTensor()
            ]),
        }
    else:
        print('no legal data augmentation is selected')
        return -1
    return data_transforms
