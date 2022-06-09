import torch
import glob
import os
import numpy as np
from pathlib import Path
def main():
    checkpoints = glob.glob("../rcgrad_flow/models/*ckpt")
    for ckpt in checkpoints:
        fname = Path(ckpt).stem
        print(fname)
        state_dict = torch.load(ckpt, map_location=torch.device('cpu'))["state_dict"]
        conv_flow_wts = state_dict['image_encoder.conv1_flow.weight'].numpy()
        np.save(f"../rcgrad_flow/flow_wts/{fname}.npy", conv_flow_wts.flatten())

if __name__ == "__main__":
    os.makedirs("../rcgrad_flow/flow_wts/", exist_ok=True)
    main()