import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
from models.VFIformer_arch import VFIformerSmall
import datasets_haze_voc as datasets
from models.utils import coords_grid
import torchgeometry as tgm

@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    mace_list = []

    for i_batch, inputs in enumerate(tqdm(val_loader)):
        img0 = inputs['img0'].to(device)
        img1 = inputs['img1'].to(device)
        flow_gt = inputs['flow_gt'].to(device)

        # Run model forward and get four_point_predictions
        # The model should return (feat, four_point_predictions, warped_img1) for test_mode=True, outloop=0
        _, four_point_predictions, _ = model.forward(img0, img1, test_mode=True, outloop=0)

        # Use the last element in four_point_predictions for four-corner prediction
        four_point = four_point_predictions[-1]  # shape: [B, 2, 2, 2]

        B = flow_gt.shape[0]
        flow_4cor = torch.zeros_like(four_point)

        flow_4cor[:, :, 0, 0] = flow_gt[:, :, 0, 0]
        flow_4cor[:, :, 0, 1] = flow_gt[:, :, 0, -1]
        flow_4cor[:, :, 1, 0] = flow_gt[:, :, -1, 0]
        flow_4cor[:, :, 1, 1] = flow_gt[:, :, -1, -1]

        mace = torch.sum((four_point - flow_4cor) ** 2, dim=1).sqrt()
        mace_list.append(mace.mean().item())

    mace_mean = np.mean(mace_list)
    print("Validation MACE: {:.6f}".format(mace_mean))
    return mace_mean


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pth)')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        cfgs = DictConfig(OmegaConf.load(f))
    device = torch.device(f'cuda:{cfgs.gpuid}' if torch.cuda.is_available() else 'cpu')

    print("=> Loading model...")
    model = VFIformerSmall(cfgs).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['net'])

    print("=> Preparing validation data...")
    val_dataset = datasets.fetch_dataloader(cfgs)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=cfgs.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    print("=> Running evaluation...")
    evaluate(model, val_loader, device)