import subprocess
from subprocess import PIPE
from subprocess import Popen

import cv2 as  cv
import numpy as np
import os
import librosa
import skvideo.io
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from rethinking_visual_sound_localization.eval_utils import combine_heatmap_img
from rethinking_visual_sound_localization.models import RCGrad


def get_audio(input_mp4):
    command = "ffmpeg -i {0}.mp4 -ab 160k -ac 2 -ar 44100 -vn {0}.wav".format(
        input_mp4[:-4]
    )
    subprocess.call(command, shell=True)
    return "{0}.wav".format(input_mp4[:-4])


def get_fps(input_mp4):
    cap = cv.VideoCapture(input_mp4)
    fps = cap.get(cv.CAP_PROP_FPS)
    return fps


def generate_video(pred_images, video_only_file, fps):
    ffmpeg_filter = f"minterpolate='mi_mode=mci:me=hexbs:me_mode=bidir:mc_mode=aobmc:vsbmc=1:mb_size=8:search_param=32:fps=30'"
    p = Popen(
        [
            "ffmpeg",
            "-y",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-r",
            "{}".format(fps),
            "-i",
            "-",
            "-b:v",
            "10M",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-strict",
            "-2",
            "-filter:v",
            f"{ffmpeg_filter}",
            video_only_file,
        ],
        stdin=PIPE,
    )

    for im in tqdm.tqdm(pred_images):
        Image.fromarray(im).save(p.stdin, "PNG")
    p.stdin.close()
    p.wait()


def mix_audio_video(audio_file, video_only_file, audio_video_file):
    cmd = 'ffmpeg -y -i {} -i "{}" -c:v copy -c:a aac {}'.format(
        video_only_file, audio_file, audio_video_file
    )
    subprocess.call(cmd, shell=True)

def calculate_flow(img, img_next):
    # convert images to grayscale
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_next = cv.cvtColor(img_next, cv.COLOR_BGR2GRAY)
    # calculate optical flow
    flow = cv.calcOpticalFlowFarneback(img, img_next,
                            None,
                            0.5, 3, 15, 3, 5, 1.2, 0)
    # Compute the magnitude and angle of the flow vectors
    magnitude, _ = cv.cartToPolar(flow[..., 0], flow[..., 1])
    #magnitude = (np.array(magnitude)+5)/200
    return magnitude

if __name__ == "__main__":
    input_mp4 = "../../data/urbansas_filtered/video/video_8fps_audio_merged/magallanes0308_02_10_0.mp4"
    print(os.path.exists(input_mp4))
    output_mp4_prefix = "output_flow"
    fps = int(get_fps(input_mp4))

    video = skvideo.io.vread(input_mp4)
    audio_file = get_audio(input_mp4)
    audio, sr = librosa.load(audio_file, sr=16000)

    rc_grad = RCGrad()

    pred_images = []
    for i in tqdm.tqdm(range(0, video.shape[0]-1)):
        image = video[i]
        image_next = video[i+1]
        flow = cv.resize(calculate_flow(image, image_next), (224, 224))
        img = Image.fromarray(image)
        img_next = Image.fromarray(image)
        aud = audio[
            int(max(0, (i / fps) * sr - sr / 2)) : int(
                min((i / fps) * sr + sr / 2, len(audio))
            )
        ]
        vis = combine_heatmap_img(img, rc_grad.pred_audio(img, aud)*flow)
        pred_images.append(
            cv.resize(
                vis,
                dsize=(video.shape[2], video.shape[1]),
                interpolation=cv.INTER_CUBIC,
            )
        )

    generate_video(pred_images, "{}_video_only.mp4".format(output_mp4_prefix), fps)
    mix_audio_video(
        audio_file,
        "{}_video_only.mp4".format(output_mp4_prefix),
        "{}_mix_audio.mp4".format(output_mp4_prefix),
    )

