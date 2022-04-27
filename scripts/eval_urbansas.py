import tqdm

from rethinking_visual_sound_localization.data import UrbansasDataset
from rethinking_visual_sound_localization.eval_utils import compute_metrics
from rethinking_visual_sound_localization.models import CLIPTran
from rethinking_visual_sound_localization.models import RCGrad

if __name__ == "__main__":
    # Download Flickr_SoundNet https://github.com/ardasnck/learning_to_localize_sound_source#preparation as data_root
    urbansas_dataset = UrbansasDataset(
        data_root = "../urbansas_filtered/"
    )

    rc_grad = RCGrad()
    preds_rc_grad = []
    for ft, img, audio, gt_map in tqdm.tqdm(urbansas_dataset):
        preds_rc_grad.append((ft, rc_grad.pred_audio(img, audio), gt_map))
    metrics_rc_grad = compute_metrics(preds_rc_grad)
    print(metrics_rc_grad)