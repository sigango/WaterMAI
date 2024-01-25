# coding:utf-8
# By Hoang Viet Pham, Jan. 24, 2024
# Email: pvhoang0109@gmail.com

import argparse
import logging
import os
import numpy as np

import torch
import torch.nn.functional as F
from tqdm import tqdm

from models import MSNet, RTFNet, UNet
from utils.data_loading import BasicDataset
from torch.utils.data import DataLoader
from utils.dice_score import multiclass_dice_coeff, dice_coeff, precision, recall

np.random.seed(0)
torch.manual_seed(0)

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    min_dice = 1
    total_precision = 0
    total_recall = 0
    low_dice_file = list()

    # iterate over the validation set
    with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
        for batch in tqdm(
            dataloader,
            total=num_val_batches,
            desc="Validation round",
            unit="batch",
            leave=False,
        ):
            image, mask_true, name = batch["image"], batch["mask"], batch['name']

            # move images and labels to correct device and type
            image = image.to(
                device=device, dtype=torch.float32, memory_format=torch.channels_last
            )
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert (
                    mask_true.min() >= 0 and mask_true.max() <= 1
                ), "True mask indices should be in [0, 1]"
                mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
                
                # compute the Dice score
                mask_pred = torch.squeeze(mask_pred)
                dice_score_batch = dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                if min_dice > dice_score_batch:
                    min_dice = dice_score_batch
                    low_dice_file = name

                dice_score += dice_score_batch
                total_precision += precision(
                    mask_pred, mask_true, reduce_batch_first=False
                )
                total_recall += recall(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert (
                    mask_true.min() >= 0 and mask_true.max() < net.n_classes
                ), "True mask indices should be in [0, n_classes["
                # convert to one-hot format
                mask_true = (
                    F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                )
                mask_pred = (
                    F.one_hot(mask_pred.argmax(dim=1), net.n_classes)
                    .permute(0, 3, 1, 2)
                    .float()
                )
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(
                    mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False
                )
                total_precision += precision(
                    mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False
                )
                total_recall += recall(
                    mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False
                )

    net.train()
    return (
        dice_score / max(num_val_batches, 1),
        total_precision / max(num_val_batches, 1),
        total_recall / max(num_val_batches, 1),
    )


def get_args():
    parser = argparse.ArgumentParser(description="Predict masks from input images")
    parser.add_argument(
        "--model",
        "-m",
        default="msnet",
        help="msnet, unet, rtfnet",
        required=True,
    )
    parser.add_argument(
        "--weights",
        "-w",
        default="MODEL.pth",
        metavar="FILE",
        help="Checkpoint weights",
        required=True,
    )
    parser.add_argument(
        "--input_folder",
        "-if",
        metavar="input_folder",
        default="",
        help="Filenames of input images",
        required=True,
    )
    parser.add_argument(
        "--type", "-t", type=str, default="", help="co, coir, cognirndwi (RGB+Green+NIR+NDWI))"
    )
    parser.add_argument(
        "--num_channels", "-nc", type=int, default=3, help="Number of channels in model"
    )
    parser.add_argument(
        "--n_classes", "-c", type=int, default=1, help="Number of classes"
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, default=2, help="size of each batch"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    inDir = args.input_folder

    # Unet model
    if args.model == "msnet":
        net = MSNet(n_classes=args.n_classes)
    elif args.model == "rtfnet":
        net = RTFNet(n_classes=args.n_classes)
    elif args.model == "unet":
        net = UNet(n_channels=args.num_channels, n_classes=args.n_classes, bilinear=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading model {args.model}")
    logging.info(f"Using device {device}")

    # Load weight
    net.to(device=device)
    state_dict = torch.load(args.weights, map_location=device)
    mask_values = state_dict.pop("mask_values", [0, 1])
    net.load_state_dict(state_dict)
    logging.info("Model loaded!")

    # Test dataloader
    if args.type == 'coir':
        dir_img = inDir + "/images/coir"
        dir_mask = inDir + "/labels/mask_coir"
    elif args.type == 'condwi':
        dir_img = inDir + "/images/condwi"
        dir_mask = inDir + "/labels/mask_condwi"
    elif args.type == 'cognirndwi':
        dir_img = inDir + "/images/cognirndwi"
        dir_mask = inDir + "/labels/mask_cognirndwi"
    loader_args = dict(batch_size=args.batch_size, num_workers=os.cpu_count(), pin_memory=True)
    test_set = BasicDataset(dir_img, dir_mask, 640)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    # Evaluation
    dice_score, precision, recall = evaluate(net, test_loader, device, amp=False)
    logging.info(
        f"Unet prediction:\n"
        f"\tDice score: {str(np.round(dice_score.detach().cpu().numpy(), 3))}\n"
        f"\tPrecision: {str(np.round(precision.detach().cpu().numpy(), 3))}\n"
        f"\tRecall: {str(np.round(recall.detach().cpu().numpy(), 3))}"
    )
