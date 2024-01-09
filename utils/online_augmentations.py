from typing import Any

import cv2
import torch
import numpy as np
from scipy.special import perm
from torchvision.transforms import (
    v2,
    Resize,
    ToPILImage,
    ToTensor,
    Compose,
    Grayscale,
    GaussianBlur,
)

from utils.visual_usage import patchify, unpatchify
from utils.fmix import sample_mask, FMixBase


# generate random bounding box
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int64(W * cut_rat)
    cut_h = np.int64(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def saliency_bbox(img, lam):
    size = img.size()
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # initialize OpenCV's static fine grained saliency detector and
    # compute the saliency map
    temp_img = img.cpu().numpy().transpose(1, 2, 0)
    # opencv-contrib-python is needed
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    _, saliencyMap = saliency.computeSaliency(temp_img)
    saliencyMap = (saliencyMap * 255).astype("uint8")

    maximum_indices = np.unravel_index(
        np.argmax(saliencyMap, axis=None), saliencyMap.shape
    )
    x = maximum_indices[0]
    y = maximum_indices[1]

    bbx1 = np.clip(x - cut_w // 2, 0, W)
    bby1 = np.clip(y - cut_h // 2, 0, H)
    bbx2 = np.clip(x + cut_w // 2, 0, W)
    bby2 = np.clip(y + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


# augmentation SAMPLE
class Cutout(object):
    def __init__(self, alpha=2, shuffle_p=1.0, class_num=2, batch_size=4, device="cpu"):
        self.alpha = alpha
        self.class_num = class_num
        self.batch_size = batch_size
        self.p = shuffle_p
        self.device = torch.device(device)

    def __call__(self, inputs, labels, act=True):
        labels = torch.eye(self.class_num).to(self.device)[
            labels, :
        ]  # one-hot hard label
        ori_inputs = inputs.clone().to(self.device)  # duplicate inputs for ori inputs
        cutout_inputs = inputs.clone().to(self.device)  # duplicate inputs for outputs
        lam_list = []  # a list to record operating ratio

        for i in range(self.batch_size):
            if np.random.randint(0, 101) > 100 * self.p or (not act):
                # trigger the augmentation operation
                lam_list.append(-1)
                continue

            lam = np.random.beta(self.alpha, self.alpha)
            bbx1, bby1, bbx2, bby2 = rand_bbox(
                ori_inputs.size(), lam
            )  # get random bbox

            cutout_inputs[i, :, bbx1:bbx2, bby1:bby2] = 0

            # update the ratio of (area of ori_image on new masked image) for soft-label
            lam = 1 - (
                (bbx2 - bbx1)
                * (bby2 - bby1)
                / (ori_inputs.size()[2] * ori_inputs.size()[3])
            )
            lam_list.append(lam)

        long_label = labels.argmax(dim=1)

        # NOTICE cutout use long label and ori_crossentropy instead of soft-label and soft-label_crossentropy
        return cutout_inputs, long_label, long_label


class CutMix(object):
    def __init__(self, alpha=2, shuffle_p=1.0, class_num=2, batch_size=4, device="cpu"):
        self.alpha = alpha
        self.class_num = class_num
        self.batch_size = batch_size

        # calibrate the trigger chance of p, new ratio is the change of operation occur in each batch
        self.p = shuffle_p * (
            perm(self.batch_size, self.batch_size)
            / (
                perm(self.batch_size, self.batch_size)
                - perm(self.batch_size - 1, self.batch_size - 1)
            )
        )
        self.device = torch.device(device)

    def __call__(self, inputs, labels, act=True):
        labels = torch.eye(self.class_num).to(self.device)[
            labels, :
        ]  # one-hot hard label
        ori_inputs = inputs.clone().to(self.device)  # duplicate inputs for ori inputs
        cutmix_inputs = inputs.clone().to(self.device)  # duplicate inputs for outputs
        lam_list = []  # a list to record operating ratio
        indices = torch.randperm(self.batch_size, device=self.device)  # shuffle indices
        shuffled_inputs = inputs[indices].to(self.device)
        shuffled_labels = labels[indices].to(self.device)

        for i in range(self.batch_size):
            if np.random.randint(0, 101) > 100 * self.p or (not act):
                # trigger the augmentation operation
                lam_list.append(-1)
                continue

            lam = np.random.beta(self.alpha, self.alpha)
            bbx1, bby1, bbx2, bby2 = rand_bbox(
                ori_inputs.size(), lam
            )  # get random bbox

            cutmix_inputs[i, :, bbx1:bbx2, bby1:bby2] = shuffled_inputs[
                i, :, bbx1:bbx2, bby1:bby2
            ]

            # update the ratio of (area of ori_image on new image) for soft-label
            lam = 1 - (
                (bbx2 - bbx1)
                * (bby2 - bby1)
                / (ori_inputs.size()[2] * ori_inputs.size()[3])
            )
            lam_list.append(lam)
            labels[i] = labels[i] * lam + shuffled_labels[i] * (1 - lam)

        long_label = labels.argmax(dim=1)
        return cutmix_inputs, labels, long_label


class Mixup(object):
    def __init__(self, alpha=2, shuffle_p=1.0, class_num=2, batch_size=4, device="cpu"):
        self.alpha = alpha
        self.class_num = class_num
        self.batch_size = batch_size
        # calibrate the trigger chance of p, new ratio is the change of operation occur in each batch
        self.p = shuffle_p * (
            perm(self.batch_size, self.batch_size)
            / (
                perm(self.batch_size, self.batch_size)
                - perm(self.batch_size - 1, self.batch_size - 1)
            )
        )
        self.device = torch.device(device)

    def __call__(self, inputs, labels, act=True):
        ori_labels = labels.clone().to(self.device)
        labels = torch.eye(self.class_num).to(self.device)[
            labels, :
        ]  # one-hot hard label
        ori_inputs = inputs.clone().to(self.device)  # duplicate inputs for ori inputs
        mixup_inputs = inputs.clone().to(self.device)  # duplicate inputs for outputs
        lam_list = []  # a list to record operating ratio
        # 按照indices的顺序更换同一个batch里面inputs和labels的顺序
        indices = torch.randperm(self.batch_size, device=self.device)  # shuffle indices
        shuffled_inputs = inputs[indices].to(self.device)
        shuffled_labels = labels[indices].to(self.device)

        for i in range(self.batch_size):
            if np.random.randint(0, 101) > 100 * self.p or (not act):
                # trigger the augmentation operation
                lam_list.append(-1)
                continue

            lam = np.random.beta(self.alpha, self.alpha)
            lam_list.append(lam)
            # shape: batch_size, 3, 384, 384
            mixup_inputs[i] = ori_inputs[i] * lam + shuffled_inputs[i] * (1 - lam)
            labels[i] = labels[i] * lam + shuffled_labels[i] * (1 - lam)

        return mixup_inputs, labels, ori_labels


class SaliencyMix(object):
    def __init__(self, alpha=1, shuffle_p=0.5, class_num=2, batch_size=4, device="cpu"):
        # ori batch_size=128
        self.alpha = alpha
        self.class_num = class_num
        self.batch_size = batch_size
        # calibrate the trigger chance of p, new ratio is the change of operation occur in each batch
        self.p = shuffle_p
        self.device = torch.device(device)

    def __call__(self, inputs, labels, act=True):
        labels = torch.eye(self.class_num).to(self.device)[
            labels, :
        ]  # one-hot hard label
        ori_inputs = inputs.clone().to(self.device)  # duplicate inputs for ori inputs
        saliencymix_inputs = inputs.clone().to(
            self.device
        )  # duplicate inputs for outputs
        lam_list = []  # a list to record operating ratio
        indices = torch.randperm(self.batch_size, device=self.device)  # shuffle indices
        shuffled_inputs = inputs[indices].to(self.device)
        shuffled_labels = labels[indices].to(self.device)

        for i in range(self.batch_size):
            if np.random.randint(0, 101) > 100 * self.p or (not act) or self.alpha <= 0:
                # trigger the augmentation operation
                lam_list.append(-1)
                continue

            lam = np.random.beta(self.alpha, self.alpha)
            bbx1, bby1, bbx2, bby2 = saliency_bbox(
                shuffled_inputs[i], lam
            )  # get random bbox

            saliencymix_inputs[i, :, bbx1:bbx2, bby1:bby2] = shuffled_inputs[
                i, :, bbx1:bbx2, bby1:bby2
            ]

            # update the ratio of (area of ori_image on new image) for soft-label
            lam = 1 - (
                (bbx2 - bbx1)
                * (bby2 - bby1)
                / (ori_inputs.size()[2] * ori_inputs.size()[3])
            )
            lam_list.append(lam)
            labels[i] = labels[i] * lam + shuffled_labels[i] * (1 - lam)

        long_label = labels.argmax(dim=1)
        return saliencymix_inputs, labels, long_label


class ResizeMix(object):
    def __init__(self, shuffle_p=1.0, class_num=2, batch_size=4, device="cpu"):
        # ori batch_size=512
        self.class_num = class_num
        self.batch_size = batch_size
        # calibrate the trigger chance of p, new ratio is the change of operation occur in each batch
        self.p = shuffle_p
        self.device = torch.device(device)

    def __call__(self, inputs, labels, alpha=0.1, beta=0.8, act=True):
        labels = torch.eye(self.class_num).to(self.device)[
            labels, :
        ]  # one-hot hard label
        ori_inputs = inputs.clone().to(self.device)  # duplicate inputs for ori inputs
        resizemix_inputs = inputs.clone().to(
            self.device
        )  # duplicate inputs for outputs
        lam_list = []  # a list to record operating ratio
        indices = torch.randperm(self.batch_size, device=self.device)  # shuffle indices
        shuffled_inputs = inputs[indices].to(self.device)
        shuffled_labels = labels[indices].to(self.device)

        for i in range(self.batch_size):
            if np.random.randint(0, 101) > 100 * self.p or (not act):
                # trigger the augmentation operation
                lam_list.append(-1)
                continue

            lam = np.random.uniform(alpha, beta)
            # lam = 1 - lam
            bbx1, bby1, bbx2, bby2 = rand_bbox(
                ori_inputs.size(), lam
            )  # get random bbox

            # resizer by torchvision
            torch_resize = Resize([bbx2 - bbx1, bby2 - bby1])

            # Tensor -> PIL -> resize -> Tensor
            re_pil_image = torch_resize(ToPILImage()(shuffled_inputs[i]))
            resizemix_inputs[i, :, bbx1:bbx2, bby1:bby2] = ToTensor()(re_pil_image)

            # update the ratio of (area of ori_image on new image) for soft-label
            lam = 1 - (
                (bbx2 - bbx1)
                * (bby2 - bby1)
                / (ori_inputs.size()[2] * ori_inputs.size()[3])
            )
            lam_list.append(lam)
            labels[i] = labels[i] * lam + shuffled_labels[i] * (1 - lam)

        long_label = labels.argmax(dim=1)
        return resizemix_inputs, labels, long_label


class FMix(FMixBase):
    def __init__(
        self,
        shuffle_p=1.0,
        class_num=2,
        batch_size=4,
        device="cpu",
        decay_power=3,
        alpha=1,
        size=(32, 32),
        max_soft=0.0,
        reformulate=False,
    ):
        # ori batch_size=128
        super().__init__(decay_power, alpha, size, max_soft, reformulate)
        self.class_num = class_num
        self.batch_size = batch_size
        self.p = shuffle_p
        self.device = torch.device(device)

    def __call__(self, inputs, labels, alpha=1, act=True):
        # Sample mask and generate random permutation
        lam, mask = sample_mask(
            self.alpha, self.decay_power, self.size, self.max_soft, self.reformulate
        )
        mask = torch.from_numpy(mask).float().to(self.device)

        labels = torch.eye(self.class_num).to(self.device)[
            labels, :
        ]  # one-hot hard label
        ori_inputs = inputs.clone().to(self.device)
        fmix_inputs = inputs.clone().to(self.device)  # duplicate inputs for outputs
        lam_list = []  # a list to record operating ratio
        indices = torch.randperm(self.batch_size, device=self.device)  # shuffle indices
        shuffled_inputs = inputs[indices].to(self.device)
        shuffled_labels = labels[indices].to(self.device)

        for i in range(self.batch_size):
            if np.random.randint(0, 101) > 100 * self.p or (not act):
                # trigger the augmentation operation
                lam_list.append(-1)
                continue

            x1 = mask * ori_inputs[i]
            x2 = (1 - mask) * shuffled_inputs[i]
            fmix_inputs[i] = x1 + x2

            lam_list.append(lam)
            labels[i] = labels[i] * lam + shuffled_labels[i] * (1 - lam)

        long_label = labels.argmax(dim=1)
        # print('lam:', lam)
        return fmix_inputs, labels, long_label


def Gradient(images: torch.Tensor) -> torch.Tensor:
    """
    Args:
        images: tensor with shape (batch_size, c, h, w)

    Returns:

    """
    transform = Compose(
        (
            Grayscale(),
            GaussianBlur(kernel_size=5, sigma=1.5),
        )
    )
    blur = transform(images)
    # blur = blur[None, ...]

    gradient_x = blur - blur.roll(-1, -1)
    gradient_x[..., :, -1] = 0

    gradient_y = blur - blur.roll(-1, -2)
    gradient_y[..., -1, :] = 0

    gradient_sum = torch.sqrt(torch.pow(gradient_x, 2) + torch.pow(gradient_y, 2))

    gradient_max = torch.max(gradient_sum[..., :-1, :-1])
    gradient_min = torch.min(gradient_sum[..., :-1, :-1])
    normalized_gradient = (gradient_sum - gradient_min) / (gradient_max - gradient_min)

    gradient = normalized_gradient.repeat(1, 3, 1, 1)

    return gradient


class HOGMask:
    def __init__(self, alpha=2, shuffle_p=1.0, class_num=2, batch_size=4, device="cpu"):
        self.class_num = class_num  # classification catagory number of the task
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.alpha = alpha
        self.p = shuffle_p * (
            perm(self.batch_size, self.batch_size)
            / (
                perm(self.batch_size, self.batch_size)
                - perm(self.batch_size - 1, self.batch_size - 1)
            )
        )

    def __call__(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        fixed_position_ratio=0.5,
        patch_size=16,
        act=True,
    ):
        if np.random.randint(0, 101) > 100 * self.p or (not act):
            soft_label = torch.eye(self.class_num).to(self.device)[
                labels, :
            ]  # one-hot hard label
            return inputs, soft_label, labels

        hog_inputs = inputs.clone().to(self.device)  # duplicate inputs for outputs
        inputs = patchify(inputs, patch_size)

        batch_size, num_patches, _ = inputs.shape
        onehot_labels = torch.eye(self.class_num).to(self.device)[labels, :]

        hog_inputs = Gradient(hog_inputs)
        hog_inputs = patchify(
            hog_inputs, patch_size
        )  # (B, num_patches, patch_size**2 * 3)

        fixed_mask = (
            torch.from_numpy(
                np.random.choice(
                    [0, 1],
                    size=(batch_size, num_patches),
                    p=[1 - fixed_position_ratio, fixed_position_ratio],
                )
            )
            .unsqueeze(-1)
            .expand(batch_size, num_patches, patch_size**2 * 3)
            .to(inputs.device)
        )

        processed_inputs = torch.where(fixed_mask == 1, inputs, hog_inputs)

        inputs = unpatchify(
            processed_inputs, patch_size
        )  # restore to image size：B,3,224,224/ B,3,384,384
        long_label = onehot_labels.argmax(dim=1)  # (B, CLS)
        return inputs, onehot_labels, long_label


def compute_image_gradients(img: torch.Tensor) -> torch.Tensor:
    """Compute image gradients (dy/dx/mod) for a given image."""
    if not isinstance(img, torch.Tensor):
        raise TypeError(
            f"The `img` expects a value of <Tensor> type but got {type(img)}"
        )
    if img.ndim != 4:
        raise RuntimeError(f"The `img` expects a 4D tensor but got {img.ndim}D tensor")

    img = v2.Grayscale()(img)

    batch_size, channels, height, width = img.shape

    dy = img[..., 1:, :] - img[..., :-1, :]
    dx = img[..., :, 1:] - img[..., :, :-1]

    shapey = [batch_size, channels, 1, width]
    dy = torch.cat([dy, torch.zeros(shapey, device=img.device, dtype=img.dtype)], dim=2)
    dy = dy.view(img.shape)

    shapex = [batch_size, channels, height, 1]
    dx = torch.cat([dx, torch.zeros(shapex, device=img.device, dtype=img.dtype)], dim=3)
    dx = dx.view(img.shape)

    return torch.cat((dx, dy, torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2))), dim=1)


class HOGRush:
    def __init__(
        self, num_classes: int, batch_size: int, shuffle_p=1.0, device="cpu"
    ) -> None:
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.p = shuffle_p * (
            perm(self.batch_size, self.batch_size)
            / (
                perm(self.batch_size, self.batch_size)
                - perm(self.batch_size - 1, self.batch_size - 1)
            )
        )
        self.device = device

    def __call__(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        fixed_position_ratio=0.5,
        patch_size=16,
        act=True,
    ) -> Any:
        soft_label = torch.eye(self.num_classes).to(self.device)[labels, :]
        if np.random.randint(0, 101) > 100 * self.p or (not act):
            return inputs, soft_label, labels

        hogged_inputs = inputs.clone().to(self.device)
        inputs = patchify(inputs, patch_size)
        num_patches = inputs.shape[1]
        hogged_inputs = patchify(compute_image_gradients(hogged_inputs), patch_size)
        fixed_mask = (
            torch.from_numpy(
                np.random.choice(
                    [0, 1],
                    size=(self.batch_size, num_patches),
                    p=[1 - fixed_position_ratio, fixed_position_ratio],
                )
            )
            .unsqueeze(-1)
            .expand(self.batch_size, num_patches, patch_size**2 * 3)
            .to(inputs.device)
        )
        processed_inputs = torch.where(fixed_mask == 1, inputs, hogged_inputs)
        return unpatchify(processed_inputs, patch_size), soft_label, labels


# ask func
def get_online_augmentation(
    augmentation_name, p=0.5, class_num=2, batch_size=4, edge_size=224, device="cpu"
):
    """
    :param augmentation_name: name of data-augmentation method
    :param p: chance of triggering
    :param class_num: classification task num
    :param batch_size: batch size
    :param edge_size: edge size of img

    :param device: cpu or cuda
    """
    if augmentation_name == "Cutout":
        Augmentation = Cutout(
            alpha=2,
            shuffle_p=p,
            class_num=class_num,
            batch_size=batch_size,
            device=device,
        )
        return Augmentation

    elif augmentation_name == "CutMix":
        Augmentation = CutMix(
            alpha=2,
            shuffle_p=p,
            class_num=class_num,
            batch_size=batch_size,
            device=device,
        )
        return Augmentation

    elif augmentation_name == "Mixup":
        Augmentation = Mixup(
            alpha=2,
            shuffle_p=p,
            class_num=class_num,
            batch_size=batch_size,
            device=device,
        )
        return Augmentation

    elif augmentation_name == "SaliencyMix":
        Augmentation = SaliencyMix(
            alpha=1,
            shuffle_p=p,
            class_num=class_num,
            batch_size=batch_size,
            device=device,
        )
        return Augmentation

    elif augmentation_name == "ResizeMix":
        Augmentation = ResizeMix(
            shuffle_p=p, class_num=class_num, batch_size=batch_size, device=device
        )
        return Augmentation

    elif augmentation_name == "FMix":
        # FMIX p=1.0 beacuse the chance of trigger is determined inside its own design
        Augmentation = FMix(
            shuffle_p=1.0,
            class_num=class_num,
            batch_size=batch_size,
            device=device,
            size=(edge_size, edge_size),
        )
        return Augmentation

    elif augmentation_name == "PuzzleMix":
        return None

    elif augmentation_name == "CoMix":
        # TODO CoMix
        return None

    elif augmentation_name == "RandomMix":
        # TODO RandomMix
        return None

    elif augmentation_name == "HOGMask":
        Augmentation = HOGMask(
            alpha=2,
            shuffle_p=p,
            class_num=class_num,
            batch_size=batch_size,
            device=device,
        )
        return Augmentation

    elif augmentation_name == "HOGRush":
        Augmentation = HOGRush(
            num_classes=class_num,
            batch_size=batch_size,
            shuffle_p=p,
            device=device,
        )
        return Augmentation

    else:
        print("no valid counterparts augmentation selected")
        return None
