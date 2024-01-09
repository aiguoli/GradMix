from __future__ import print_function, division

import copy
from dataclasses import asdict
from datetime import datetime
import json
import time
import os
import numpy as np
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from torchinfo import summary
import matplotlib
import wandb
from config import Config

from logger import get_logger
from utils.data_augmentation import data_augmentation
from utils.SoftCrossEntropyLoss import SoftlabelCrossEntropy
from utils.online_augmentations import get_online_augmentation
from utils.visual_usage import visualize_check, check_SAA
from utils.tools import setup_seed, del_file, FixStateDict
from utils.schedulers import patch_scheduler, ratio_scheduler
from models.getmodel import get_model
from models.GetPromptModel import build_promptmodel


# Training Script
def better_performance(
    temp_acc, temp_vac, best_acc, best_vac
):  # determin which epoch have the best model
    if temp_vac >= best_vac and temp_acc >= best_acc:
        return True
    elif temp_vac > best_vac:
        return True
    else:
        return False


def train_model(
    model: nn.Module,
    dataloaders: dict,
    criterion,
    optimizer,
    class_names,
    dataset_sizes,
    Augmentation=None,
    fix_position_ratio_scheduler=None,
    puzzle_patch_size_scheduler=None,
    edge_size=384,
    model_idx=None,
    num_epochs=25,
    intake_epochs=0,
    check_minibatch=100,
    scheduler=None,
    device=None,
    draw_path="../imagingresults",
    enable_attention_check=False,
    enable_visualize_check=False,
    enable_sam=False,
    writer=None,
    logger=None,
):
    """
    Training iteration
    :param model: model object
    :param dataloaders: 2 dataloader(train and val) dict
    :param criterion: loss func obj
    :param optimizer: optimizer obj
    :param class_names: The name of classes for priting
    :param dataset_sizes: size of datasets
    :param Augmentation: Online augmentation
    :param fix_position_ratio_scheduler: Online augmentation fix_position_ratio_scheduler
    :param puzzle_patch_size_scheduler: Online augmentation puzzle_patch_size_scheduler
    :param edge_size: image size for the input image
    :param model_idx: model idx for the getting pre-setted model
    :param num_epochs: total training epochs
    :param intake_epochs: number of skip over epochs when choosing the best model
    :param check_minibatch: number of skip over minibatch in calculating the criteria's results etc.
    :param scheduler: scheduler is an LR scheduler object from torch.optim.lr_scheduler.
    :param device: cpu/gpu object
    :param draw_path: path folder for output pic
    :param enable_attention_check: use attention_check to show the pics of models' attention areas
    :param enable_visualize_check: use visualize_check to show the pics
    :param enable_sam: use SAM training strategy
    :param writer: attach the records to the tensorboard backend
    """

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    since = time.time()

    # for saving the best model state dict
    best_model_wts = copy.deepcopy(model.state_dict())  # deepcopy
    # initial an empty dict
    json_log = {}

    # initial best performance
    best_acc = 0.0
    best_vac = 0.0
    temp_acc = 0.0
    temp_vac = 0.0
    best_epoch_idx = 1

    epoch_loss = 0.0  # initial value for loss-drive

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)

        # record json log, initially empty
        json_log[str(epoch + 1)] = {}

        # Each epoch has a training and validation phase
        for phase in ("train", "val"):  # alternatively train/val
            index = 0
            check_index = (
                -1
            )  # set a visulize check at the end of each epoch's train and val
            model_time = time.time()

            # initiate the empty log dict
            log_dict = {}
            for cls_idx in range(len(class_names)):
                # only float type is allowed in json, set to float inside
                log_dict[class_names[cls_idx]] = {
                    "tp": 0.0,
                    "tn": 0.0,
                    "fp": 0.0,
                    "fn": 0.0,
                }

            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # criterias, initially empty
            running_loss = 0.0
            log_running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[
                phase
            ]:  # use different dataloder in different phase
                inputs = inputs.to(device)  # print('inputs[0]',type(inputs[0]))
                # NOTICE in CLS task the labels' type is long tensor([B])，not one-hot ([B,CLS])
                labels = labels.to(device)

                # Online Augmentations on device
                if Augmentation is not None:
                    if phase == "train":
                        if (
                            fix_position_ratio_scheduler is not None
                            and puzzle_patch_size_scheduler is not None
                        ):
                            fix_position_ratio = fix_position_ratio_scheduler(
                                epoch, epoch_loss
                            )
                            puzzle_patch_size = puzzle_patch_size_scheduler(
                                epoch, epoch_loss
                            )

                            inputs, labels, GT_long_labels = Augmentation(
                                inputs, labels, fix_position_ratio, puzzle_patch_size
                            )
                        # Counterpart augmentations
                        else:
                            inputs, labels, GT_long_labels = Augmentation(
                                inputs, labels
                            )

                    else:  # Val
                        inputs, labels, GT_long_labels = Augmentation(
                            inputs, labels, act=False
                        )
                else:
                    GT_long_labels = labels  # store ori_label on CPU

                # zero the parameter gradients
                if not enable_sam:
                    optimizer.zero_grad()

                # forward
                # track grad if only in train!
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)  # pred outputs of confidence: [B,CLS]
                    _, preds = torch.max(outputs, 1)  # idx outputs: [B] each is a idx
                    soft_labels = torch.eye(len(class_names)).to(labels.device)[
                        GT_long_labels, :
                    ]
                    loss = criterion(
                        outputs, labels
                    )  # cross entrphy of one-hot outputs: [B,CLS] and idx label [B]

                    # backward + optimize only if in training phase
                    if phase == "train":
                        if enable_sam:
                            loss.backward()
                            # first forward-backward pass
                            optimizer.first_step(zero_grad=True)

                            # second forward-backward pass
                            loss2 = criterion(
                                model(inputs), labels
                            )  # SAM need another model(inputs)
                            loss2.backward()  # make sure to do a full forward pass when using SAM
                            optimizer.second_step(zero_grad=True)
                        else:
                            loss.backward()
                            optimizer.step()

                # log criterias: update
                log_running_loss += loss.item()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds.cpu() == GT_long_labels.cpu().data)

                # Compute precision and recall for each class.
                for cls_idx in range(len(class_names)):
                    tp = np.dot(
                        (GT_long_labels.cpu().data == cls_idx).numpy().astype(int),
                        (preds == cls_idx).cpu().numpy().astype(int),
                    )
                    tn = np.dot(
                        (GT_long_labels.cpu().data != cls_idx).numpy().astype(int),
                        (preds != cls_idx).cpu().numpy().astype(int),
                    )

                    fp = np.sum((preds == cls_idx).cpu().numpy()) - tp

                    fn = np.sum((GT_long_labels.cpu().data == cls_idx).numpy()) - tp

                    # log_dict[cls_idx] = {'tp': 0.0, 'tn': 0.0, 'fp': 0.0, 'fn': 0.0} set to float inside
                    log_dict[class_names[cls_idx]]["tp"] += tp
                    log_dict[class_names[cls_idx]]["tn"] += tn
                    log_dict[class_names[cls_idx]]["fp"] += fp
                    log_dict[class_names[cls_idx]]["fn"] += fn

                # attach the records to the tensorboard backend
                if writer is not None:
                    # ...log the running loss
                    writer.add_scalar(
                        phase + " minibatch loss",
                        float(loss.item()),
                        epoch * len(dataloaders[phase]) + index,
                    )
                    writer.add_scalar(
                        phase + " minibatch ACC",
                        float(
                            torch.sum(preds.cpu() == GT_long_labels.cpu().data)
                            / inputs.size(0)
                        ),
                        epoch * len(dataloaders[phase]) + index,
                    )

                # at the checking time now
                if index % check_minibatch == check_minibatch - 1:
                    model_time = time.time() - model_time

                    check_index = index // check_minibatch + 1

                    epoch_idx = epoch + 1
                    logger.info(
                        "Epoch: {} phase: {}, index of {} minibatch: {}, time used: {}".format(
                            epoch_idx, phase, check_minibatch, check_index, model_time
                        )
                    ),

                    logger.info(
                        "minibatch AVG loss: {}".format(
                            float(log_running_loss) / check_minibatch
                        )
                    )

                    if enable_visualize_check:
                        visualize_check(
                            inputs,
                            GT_long_labels,
                            model,
                            class_names,
                            num_images=3,
                            pic_name="Visual_"
                            + phase
                            + "_E_"
                            + str(epoch_idx)
                            + "_I_"
                            + str(index + 1),
                            draw_path=draw_path,
                            writer=writer,
                        )

                    if enable_attention_check:
                        try:
                            check_SAA(
                                inputs,
                                GT_long_labels,
                                model,
                                model_idx,
                                edge_size,
                                class_names,
                                num_images=1,
                                pic_name="GradCAM_"
                                + phase
                                + "_E_"
                                + str(epoch_idx)
                                + "_I_"
                                + str(index + 1),
                                draw_path=draw_path,
                                writer=writer,
                            )
                        except Exception as e:
                            logger.info(
                                "When checking attention, an error occur: {}".format(e)
                            )
                            logger.info(
                                "model: {} with edge_size {} is not supported yes".format(
                                    model_idx, edge_size
                                )
                            )
                    else:
                        pass

                    model_time = time.time()
                    log_running_loss = 0.0

                index += 1

            if phase == "train":
                if scheduler is not None:  # lr scheduler: update
                    scheduler.step()

            # at the last of train/val in each epoch, if no check has been triggered
            if check_index == -1:
                epoch_idx = epoch + 1
                if enable_visualize_check:
                    visualize_check(
                        inputs,
                        GT_long_labels,
                        model,
                        class_names,
                        num_images=3,
                        pic_name="Visual_" + phase + "_E_" + str(epoch_idx),
                        draw_path=draw_path,
                        writer=writer,
                    )

                if enable_attention_check:
                    try:
                        check_SAA(
                            inputs,
                            GT_long_labels,
                            model,
                            model_idx,
                            edge_size,
                            class_names,
                            num_images=1,
                            pic_name="GradCAM_" + phase + "_E_" + str(epoch_idx),
                            draw_path=draw_path,
                            writer=writer,
                        )
                    except:
                        print(
                            "model:",
                            model_idx,
                            " with edge_size",
                            edge_size,
                            "is not supported yet",
                        )
                else:
                    pass

            # log criterias: print
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase] * 100
            logger.info(
                "Epoch: {}  {} \n Loss: {:.4f}  Acc: {:.4f}".format(
                    epoch + 1, phase, epoch_loss, epoch_acc
                )
            )
            wandb.log(
                {
                    "epoch": epoch,
                    "{}_loss".format(phase): epoch_loss,
                    "{}_acc".format(phase): epoch_acc,
                }
            )

            if (
                phase == "train"
                and fix_position_ratio_scheduler is not None
                and puzzle_patch_size_scheduler is not None
            ):
                logger.info(
                    "Epoch: {}, Fix_position_ratio: {}, Puzzle_patch_size: "
                    "{}".format(epoch + 1, fix_position_ratio, puzzle_patch_size)
                )

            # attach the records to the tensorboard backend
            if writer is not None:
                # ...log the running loss
                writer.add_scalar(phase + " loss", float(epoch_loss), epoch + 1)
                writer.add_scalar(phase + " ACC", float(epoch_acc), epoch + 1)

            # calculating the confusion matrix
            for class_name in class_names:
                tp = log_dict[class_name]["tp"]
                tn = log_dict[class_name]["tn"]
                fp = log_dict[class_name]["fp"]
                fn = log_dict[class_name]["fn"]
                tp_plus_fp = tp + fp
                tp_plus_fn = tp + fn
                fp_plus_tn = fp + tn
                fn_plus_tn = fn + tn

                # precision
                if tp_plus_fp == 0:
                    precision = 0
                else:
                    precision = float(tp) / tp_plus_fp * 100
                # recall
                if tp_plus_fn == 0:
                    recall = 0
                else:
                    recall = float(tp) / tp_plus_fn * 100

                # TPR (sensitivity)
                TPR = recall

                # TNR (specificity)
                # FPR
                if fp_plus_tn == 0:
                    TNR = 0
                    FPR = 0
                else:
                    TNR = tn / fp_plus_tn * 100
                    FPR = fp / fp_plus_tn * 100

                # NPV
                if fn_plus_tn == 0:
                    NPV = 0
                else:
                    NPV = tn / fn_plus_tn * 100

                logger.info(
                    "{} precision: {:.4f}  recall: {:.4f}".format(
                        class_name, precision, recall
                    )
                )
                logger.info(
                    "{} sensitivity: {:.4f}  specificity: {:.4f}".format(
                        class_name, TPR, TNR
                    )
                )
                logger.info("{} FPR: {:.4f}  NPV: {:.4f}".format(class_name, FPR, NPV))
                if (precision + recall) != 0:
                    logger.info(
                        "{} F1: {:.4f}".format(
                            class_name, 2 * precision * recall / (precision + recall)
                        )
                    )
                logger.info("{} TP: {}".format(class_name, tp))
                logger.info("{} TN: {}".format(class_name, tn))
                logger.info("{} FP: {}".format(class_name, fp))
                logger.info("{} FN: {}".format(class_name, fn))
                # attach the records to the tensorboard backend
                if writer is not None:
                    # ...log the running loss
                    writer.add_scalar(
                        phase + "   " + class_name + " precision",
                        precision,
                        epoch + 1,
                    )
                    writer.add_scalar(
                        phase + "   " + class_name + " recall",
                        recall,
                        epoch + 1,
                    )

            # json log: update
            json_log[str(epoch + 1)][phase] = log_dict

            if phase == "val":
                temp_vac = epoch_acc
            else:
                temp_acc = epoch_acc  # not useful actually

            # deep copy the model
            if (
                phase == "val"
                and better_performance(temp_acc, temp_vac, best_acc, best_vac)
                and epoch >= intake_epochs
            ):
                # what is better? we now use the wildly used method only
                best_epoch_idx = epoch + 1
                best_acc = temp_acc
                best_vac = temp_vac
                best_model_wts = copy.deepcopy(model.state_dict())
                best_log_dic = log_dict

        if epoch % 5 == 0:
            check_path = "checkpoints/{}_{}_{}".format(
                datetime.now().strftime("%Y_%m_%d_%H_%M"),
                config.augmentation_name,
                os.path.basename(config.dataroot),
            )
            os.mkdir(check_path)
            torch.save(model.state_dict(), check_path + "/{}.pth".format(epoch))
            logger.info("Epoch {} checkpoint has been saved!".format)

    time_elapsed = time.time() - since
    logger.info(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    logger.info("Best epoch idx: {}".format(best_epoch_idx))
    logger.info("Best epoch train Acc: {:4f}".format(best_acc))
    logger.info("Best epoch val Acc: {:4f}".format(best_vac))
    for class_name in class_names:
        tp = best_log_dic[class_name]["tp"]
        tn = best_log_dic[class_name]["tn"]
        fp = best_log_dic[class_name]["fp"]
        fn = best_log_dic[class_name]["fn"]
        tp_plus_fp = tp + fp
        tp_plus_fn = tp + fn
        fp_plus_tn = fp + tn
        fn_plus_tn = fn + tn

        # precision
        if tp_plus_fp == 0:
            precision = 0
        else:
            precision = float(tp) / tp_plus_fp * 100
        # recall
        if tp_plus_fn == 0:
            recall = 0
        else:
            recall = float(tp) / tp_plus_fn * 100

        # TPR (sensitivity)
        TPR = recall

        # TNR (specificity)
        # FPR
        if fp_plus_tn == 0:
            TNR = 0
            FPR = 0
        else:
            TNR = tn / fp_plus_tn * 100
            FPR = fp / fp_plus_tn * 100

        # NPV
        if fn_plus_tn == 0:
            NPV = 0
        else:
            NPV = tn / fn_plus_tn * 100

        logger.info(
            "{} precision: {:.4f}  recall: {:.4f}".format(
                class_names[cls_idx], precision, recall
            )
        )
        logger.info(
            "{} sensitivity: {:.4f}  specificity: {:.4f}".format(
                class_names[cls_idx], TPR, TNR
            )
        )
        logger.info(
            "{} FPR: {:.4f}  NPV: {:.4f}".format(class_names[cls_idx], FPR, NPV)
        )
        logger.info(
            "{} F1: {:.4f}".format(
                class_name, 2 * precision * recall / (precision + recall)
            )
        )

    # attach the records to the tensorboard backend
    if writer is not None:
        writer.close()

    # load best model weights as final model training result
    model.load_state_dict(best_model_wts)
    # save json_log  indent=2 for better view
    json.dump(
        json_log,
        open(os.path.join(draw_path, model_idx + "_log.json"), "w"),
        ensure_ascii=False,
        indent=2,
    )
    return model


def main(config: Config):
    if config.paint:
        # use Agg kernal, not painting in the front-desk

        matplotlib.use("Agg")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    enable_tensorboard = config.enable_tensorboard  # True
    enable_attention_check = config.enable_attention_check  # False   'CAM' 'SAA'
    enable_visualize_check = config.enable_visualize_check  # False

    enable_sam = config.enable_sam  # False

    data_augmentation_mode = config.data_augmentation_mode  # 0

    linearprobing = config.linearprobing  # False

    Pre_Trained_model_path = config.Pre_Trained_model_path  # None
    Prompt_state_path = config.Prompt_state_path  # None

    # Prompt
    PromptTuning = config.PromptTuning  # None  "Deep" / "Shallow"
    Prompt_Token_num = config.Prompt_Token_num  # 20
    PromptUnFreeze = config.PromptUnFreeze  # False

    gpu_idx = config.gpu_idx  # GPU idx start with0, -1 to use multipel GPU

    # model info
    model_idx = (
        config.model_idx
    )  # the model we are going to use. by the format of Model_size_other_info
    # structural parameter
    drop_rate = config.drop_rate
    attn_drop_rate = config.attn_drop_rate
    drop_path_rate = config.drop_path_rate
    use_cls_token = False if config.cls_token_off else True
    use_pos_embedding = False if config.pos_embedding_off else True
    use_att_module = None if config.att_module == "None" else config.att_module

    # pretrained_backbone
    pretrained_backbone = False if config.backbone_PT_off else True

    # classification required number of your dataset
    num_classes = config.num_classes  # default 0 for auto-fit
    # image size for the input image
    edge_size = config.edge_size  # 224 384 1000

    # batch info
    batch_size = config.batch_size  # 8
    num_workers = config.num_workers  # main training num_workers 4

    num_epochs = config.num_epochs  # 50
    intake_epochs = config.intake_epochs  # 0
    check_minibatch = (
        config.check_minibatch
        if config.check_minibatch is not None
        else 400 // batch_size
    )

    lr = config.lr
    lrf = config.lrf

    opt_name = config.opt_name  # 'Adam'

    # PATH info
    draw_root = config.draw_root
    model_path = config.model_path
    dataroot = config.dataroot

    draw_path = os.path.join(
        draw_root, "CLS_" + model_idx
    )  # PC is for the pancreatic cancer
    save_model_path = os.path.join(
        model_path,
        "{}_{}_{}_{}_{}.pth".format(
            config.augmentation_name,
            os.path.basename(config.dataroot),
            config.ratio_strategy or config.fix_position_ratio,
            config.patch_strategy or config.fix_patch_size,
            config.edge_size,
        ),
    )

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if os.path.exists(draw_path):
        del_file(
            draw_path
        )  # fixme clear the output folder, NOTICE this may be DANGEROUS
    else:
        os.makedirs(draw_path)

    # Train Augmentation
    augmentation_name = config.augmentation_name  # None

    # logger
    logger = get_logger(
        "{}_{}_{}".format(
            config.augmentation_name, config.model_idx, config.dataroot.split("/")[-1]
        )
    )
    logger.info("{:-^50s}".format(" Config "))
    for arg in vars(config):
        logger.info("{}: {}".format(arg, getattr(config, arg)))
    logger.info("-" * 50)

    # Data Augmentation
    data_transforms = data_augmentation(data_augmentation_mode, edge_size=edge_size)

    datasets = {
        x: torchvision.datasets.ImageFolder(
            os.path.join(dataroot, x), data_transforms[x]
        )
        for x in ["train", "val"]
    }  # 2 dataset obj is prepared here and combine together
    dataset_sizes = {
        x: len(datasets[x]) for x in ["train", "val"]
    }  # size of each dataset

    dataloaders = {
        "train": torch.utils.data.DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        ),  # colab suggest 2
        "val": torch.utils.data.DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers // 4 + 1,
            drop_last=True,
        ),
    }

    class_names = [
        d.name for d in os.scandir(os.path.join(dataroot, "train")) if d.is_dir()
    ]
    class_names.sort()
    if num_classes == 0:
        logger.info(
            "class_names: {}".format(" ".join(class_name for class_name in class_names))
        )
        num_classes = len(class_names)
    else:
        if len(class_names) == num_classes:
            logger.info(
                "class_names: {}".format(
                    " ".join(class_name for class_name in class_names)
                )
            )
        else:
            logger.info(
                "classfication number of the model mismatch the dataset requirement of: {}".format(
                    len(class_names)
                )
            )
            return -1
    print("{:*^50s}".format(" Settings "))
    logger.info(config)

    # start tensorboard backend
    if enable_tensorboard:
        writer = SummaryWriter(draw_path)
    else:
        writer = None
    # if u run locally
    # nohup tensorboard --logdir=/home/MSHT/runs --host=0.0.0.0 --port=7777 &
    # tensorboard --logdir=/home/ZTY/runs --host=0.0.0.0 --port=7777

    if gpu_idx == -1:  # use all cards
        if torch.cuda.device_count() > 1:
            print("Use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            gpu_use = gpu_idx
        else:
            print("we dont have more GPU idx here, try to use gpu_idx=0")
            try:
                os.environ[
                    "CUDA_VISIBLE_DEVICES"
                ] = "0"  # setting k for: only card idx k is sighted for this code
                gpu_use = 0
            except:
                print("GPU distributing ERRO occur use CPU instead")
                gpu_use = "cpu"

    else:
        # Decide which device we want to run on
        try:
            # setting k for: only card idx k is sighted for this code
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            gpu_use = gpu_idx
        except:
            logger.info("we dont have that GPU idx here, try to use gpu_idx=0")
            try:
                # setting 0 for: only card idx 0 is sighted for this code
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                gpu_use = 0
            except:
                logger.info("GPU distributing ERRO occur use CPU instead")
                gpu_use = "cpu"

    # device enviorment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get model
    if PromptTuning is not None:
        print("PromptTuning of ", model_idx)
        print("Prompt VPT type:", PromptTuning)
        if Pre_Trained_model_path is not None and os.path.exists(
            Pre_Trained_model_path
        ):
            base_state_dict = torch.load(Pre_Trained_model_path)
        else:
            base_state_dict = "timm"
            print("base_state_dict of timm")

        if Prompt_state_path is not None and os.path.exists(Prompt_state_path):
            prompt_state_dict = torch.load(Prompt_state_path)
        else:
            prompt_state_dict = None
            print("prompt_state of None")

        model = build_promptmodel(
            num_classes,
            edge_size,
            model_idx,
            Prompt_Token_num=Prompt_Token_num,
            VPT_type=PromptTuning,
            prompt_state_dict=prompt_state_dict,
            base_state_dict=base_state_dict,
        )
        if PromptUnFreeze:
            model.UnFreeze()
            print("prompt tuning will all parameaters un-freezed")
    else:
        # get model: randomly initiate model, except the backbone CNN(when pretrained_backbone is True)
        model = get_model(
            num_classes,
            edge_size,
            model_idx,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            pretrained_backbone,
            use_cls_token,
            use_pos_embedding,
            use_att_module,
        )

        # Manually get the model pretrained on the Imagenet1000
        if Pre_Trained_model_path is not None:
            if os.path.exists(Pre_Trained_model_path):
                state_dict = FixStateDict(
                    torch.load(Pre_Trained_model_path), remove_key_head="head"
                )
                model.load_state_dict(state_dict, False)
                logger.info("pretrain model loaded")
            else:
                logger.info(
                    "Pre_Trained_model_path:" + Pre_Trained_model_path,
                    " is NOT avaliable!!!!\n",
                )
                raise  # print('we ignore this with a new start up')

        if linearprobing:
            # Only tuning the last FC layer for CLS task
            module_all = 0
            for child in model.children():  # find all nn.modules
                module_all += 1

            for param in model.parameters():  # freeze all parameters
                param.requires_grad = False

            for module_idx, child in enumerate(model.children()):
                if (
                    module_idx == module_all
                ):  # Unfreeze the parameters of the last FC layer
                    for param in child.parameters():
                        param.requires_grad = True

    logger.info("GPU: {}".format(gpu_use))

    if gpu_use == -1:
        model = nn.DataParallel(model)

    model.to(device)

    try:
        summary(
            model, input_size=(3, edge_size, edge_size)
        )  # should be after .to(device)
    except:
        pass

    logger.info("model: {}".format(model_idx))

    # Augmentation
    Augmentation = get_online_augmentation(
        augmentation_name,
        p=0.5,
        class_num=num_classes,
        batch_size=batch_size,
        edge_size=edge_size,
        device=device,
    )

    if augmentation_name != "HOGMask" and augmentation_name != "HOGRush":
        fix_position_ratio_scheduler = None
        puzzle_patch_size_scheduler = None
    else:
        # setting puzzle_patch_size and fix_position_ratio schedulers
        # 似乎是逐渐变小的
        fix_position_ratio_scheduler = ratio_scheduler(
            total_epoches=num_epochs,
            warmup_epochs=0,
            basic_ratio=0.5,
            strategy=config.ratio_strategy,  # 'linear'
            fix_position_ratio=config.fix_position_ratio,
            threshold=config.loss_drive_threshold,
        )

        puzzle_patch_size_scheduler = patch_scheduler(
            total_epoches=num_epochs,
            warmup_epochs=0,
            edge_size=edge_size,
            basic_patch=16,
            strategy=config.patch_strategy,
            threshold=config.loss_drive_threshold,
            fix_patch_size=config.fix_patch_size,
            patch_size_jump=config.patch_size_jump,
        )

    # Default cross entrphy of one-hot outputs: [B,CLS] and idx label [B] long tensor
    # augmentation loss is SoftlabelCrossEntropy
    criterion = (
        SoftlabelCrossEntropy()
        if Augmentation is not None and augmentation_name != "Cutout"
        else nn.CrossEntropyLoss()
    )
    # https://github.com/pytorch/pytorch/pull/61044
    # criterion = nn.CrossEntropyLoss()

    if opt_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=0.8, weight_decay=0.005
        )
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=20, gamma=0.5
        )  # 15 0.1  default SGD StepLR scheduler
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = None
    # 似乎都在说AdamW更好
    elif opt_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        scheduler = None
    else:
        raise
    logger.info("optimizer: {}".format(opt_name))

    if enable_sam:
        from utils.sam import SAM

        if opt_name == "SGD":
            base_optimizer = (
                torch.optim.SGD
            )  # define an optimizer for the "sharpness-aware" update
            optimizer = SAM(model.parameters(), base_optimizer, lr=lr, momentum=0.8)
            scheduler = None
        elif opt_name == "Adam":
            base_optimizer = (
                torch.optim.Adam
            )  # define an optimizer for the "sharpness-aware" update
            optimizer = SAM(
                model.parameters(), base_optimizer, lr=lr, weight_decay=0.01
            )
        else:
            raise ValueError("Only SGD and Adam are supported for SAM currently")

    if lrf > 0:  # use cosine learning rate schedule
        import math

        # cosine Scheduler by https://arxiv.org/pdf/1812.01187.pdf
        lf = (
            lambda x: ((1 + math.cos(x * math.pi / num_epochs)) / 2) * (1 - lrf) + lrf
        )  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # train
    model_ft = train_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        class_names,
        dataset_sizes,
        fix_position_ratio_scheduler=fix_position_ratio_scheduler,
        puzzle_patch_size_scheduler=puzzle_patch_size_scheduler,
        Augmentation=Augmentation,
        edge_size=edge_size,
        model_idx=model_idx,
        num_epochs=num_epochs,
        intake_epochs=intake_epochs,
        check_minibatch=check_minibatch,
        scheduler=scheduler,
        device=device,
        draw_path=draw_path,
        enable_attention_check=enable_attention_check,
        enable_visualize_check=enable_visualize_check,
        enable_sam=enable_sam,
        writer=writer,
        logger=logger,
    )

    # save model if its a multi-GPU model, save as a single GPU one too
    if gpu_use == -1:
        if PromptTuning is None:
            torch.save(model_ft.module.state_dict(), save_model_path)

        else:
            if PromptUnFreeze:
                torch.save(model_ft.module.state_dict(), save_model_path)
            else:
                prompt_state_dict = model_ft.module.obtain_prompt()
                # fixme maybe bug at DP module.obtain_prompt, just model.obtain_prompt is enough
                torch.save(prompt_state_dict, save_model_path)

        print(
            "model trained by multi-GPUs has its single GPU copy saved at ",
            save_model_path,
        )

    else:
        if PromptTuning is None:
            torch.save(model_ft.state_dict(), save_model_path)

        else:
            if PromptUnFreeze:
                torch.save(model_ft.state_dict(), save_model_path)
            else:
                prompt_state_dict = model_ft.obtain_prompt()
                torch.save(prompt_state_dict, save_model_path)

        print(
            "model trained by GPU (idx:" + str(gpu_use) + ") has been saved at ",
            save_model_path,
        )


if __name__ == "__main__":
    # setting up the random seed
    setup_seed(42)
    os.environ["WANDB_API_KEY"] = ""

    config = Config()

    wandb.init(
        project=config.dataroot.split("/")[-1],
        name=config.augmentation_name + "_" + config.model_idx,
        config=asdict(config),
    )
    main(config)
