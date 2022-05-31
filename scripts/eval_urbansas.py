import tqdm
import argparse
from PIL import Image
import numpy as np
import pandas as pd
import os

from rethinking_visual_sound_localization.data import UrbansasDataset
from rethinking_visual_sound_localization.eval_utils import compute_metrics, cal_CIOU
from rethinking_visual_sound_localization.models import CLIPTran
from rethinking_visual_sound_localization.models import RCGrad



def main():
    urbansas_dataset = UrbansasDataset(data_root = data_root)

    # get predictions
    if model == "rc_grad":
        rc_grad = RCGrad()
        preds = []
        for ft, img, audio, gt_map in tqdm.tqdm(urbansas_dataset):
            preds.append((ft, rc_grad.pred_audio(img, audio), gt_map))
    
    
    elif model == "flow":
        preds = []
        for ft, img, audio, gt_map in tqdm.tqdm(urbansas_dataset):
            try:
                flow = np.array(Image.open(f"{data_root}Flow/{ft}.jpg").resize((224, 224)))
            except:
                continue
            preds.append((ft, flow, gt_map))

    
    elif model == "rc_grad_flow":
        rc_grad = RCGrad()
        preds = []
        for ft, img, audio, gt_map in tqdm.tqdm(urbansas_dataset):
            try:
                flow = np.array(Image.open(f"{data_root}Flow/{ft}.jpg").resize((224, 224)))
            except:
                continue
            
            pred = rc_grad.pred_audio(img, audio)
            pred*=flow
            preds.append((ft, pred, gt_map))

    elif model == "clip_tran":
        clip_tran = CLIPTran()
        preds = []
        for ft, img, audio, gt_map in tqdm.tqdm(urbansas_dataset):
            preds.append((ft, clip_tran.pred_audio(img, audio), gt_map))

    # save image-wise cIoU
    ious = [cal_CIOU(pred, gt_map)[0] for _, pred, gt_map in preds]
    filenames = [ft for ft, _, _ in preds]
    iou_df = pd.DataFrame()
    iou_df["filename"] = filenames
    iou_df["iou"] = ious
    iou_df.to_csv(f"evaluation/{model}.csv", index=None)

    # compute metrics
    metrics = compute_metrics(preds)
    print(metrics)  



if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='evaluates ssl on urbansas')
    parser.add_argument('-filtered', '--f', action='store_true',
                        help='The filtered version of the dataset will be used if the argument is passed')
    parser.add_argument("-model", action = "store", default="rc_grad")

    filtered = parser.parse_args().f    
    model = parser.parse_args().model 

    dataset = "urbansas"
    if filtered:
        dataset = "urbansas_filtered"
    data_root = f"../{dataset}/"

    print(f"Using model - {model}")
    print(f"Dataset - {dataset}")

    # setup evaluation directory
    if not os.path.isdir("evaluation/"):
        os.mkdir("evaluation/")

    main()

