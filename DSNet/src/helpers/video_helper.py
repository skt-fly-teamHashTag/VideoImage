from os import PathLike
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from numpy import linalg
from torch import nn
from torchvision import transforms, models

import time

from kts.cpd_auto import cpd_auto


class FeatureExtractor(object):
    def __init__(self):
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model = models.googlenet(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.model = self.model.cuda().eval()

    def run(self, img: np.ndarray) -> np.ndarray:
        img = Image.fromarray(img)
        img = self.preprocess(img)
        batch = img.unsqueeze(0)
        with torch.no_grad():
            feat = self.model(batch.cuda())
            feat = feat.squeeze().cpu().numpy()

        assert feat.shape == (1024,), f'Invalid feature shape {feat.shape}: expected 1024'
        # normalize frame features
        feat /= linalg.norm(feat) + 1e-10
        return feat


class VideoPreprocessor(object):
    def __init__(self, sample_rate: int) -> None:
        self.model = FeatureExtractor()
        self.sample_rate = sample_rate

    def get_features(self, video_path: PathLike):
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # print( "before: ", cap.get(cv2.CAP_PROP_POS_MSEC)) #0.0
        # print(cap.set(cv2.CAP_PROP_POS_MSEC, 300))
        # print( "after: ", cap.get(cv2.CAP_PROP_POS_MSEC)) #10.0

        # print(f"before -capture FPS: {cap.get(cv2.CAP_PROP_FPS)}")
        # print(cap.set(cv2.CAP_PROP_FPS, cap.get(cv2.CAP_PROP_FPS)/5))
        # print(f"after -capture FPS: {cap.get(cv2.CAP_PROP_FPS)}")
        

        assert cap is not None, f'Cannot open video: {video_path}'

        features = []
        n_frames = 0

        start = time.time()
        while True:
            # ret, frame = cap.read() #read() = grab() + retrieve() 
            ret = cap.grab() 
            # ret = cap.grab()
            ret, frame = cap.retrieve()

            # print(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if not ret:
                break

            if n_frames % self.sample_rate == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                feat = self.model.run(frame)
                features.append(feat)
                 
            # print("change capture Params")
            # print("MSEC :", cap.set(cv2.CAP_PROP_POS_MSEC, 1.2)) # Current position of the video file in milliseconds. 현재 동영상 위치를 _s 로 옮긴다 
            # print("FRAMES :", cap.set(cv2.CAP_PROP_POS_FRAMES, 10)) # 0-based index of the frame to be decoded/captured next. 현재 프레임 위치를 몇 프레임 뒤로 옮긴다 
            # cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES)+1) # 프레임마다 10프레임씩 건너뛴다 

            n_frames += 1

        end = time.time()
        print(f"cap time: {(end-start)//60}min {(end-start)%60}s")
        cap.release()
        

        features = np.array(features)
        return n_frames, features, fps

    def kts(self, n_frames, features):
        seq_len = len(features)
        picks = np.arange(0, seq_len) * self.sample_rate

        # compute change points using KTS
        max_ncp = seq_len -1
        kernel = np.matmul(features, features.T)
        change_points, _ = cpd_auto(kernel, max_ncp, 1, verbose=False)
        change_points *= self.sample_rate
        change_points = np.hstack((0, change_points, n_frames))
        begin_frames = change_points[:-1]
        end_frames = change_points[1:]
        change_points = np.vstack((begin_frames, end_frames - 1)).T

        n_frame_per_seg = end_frames - begin_frames
        return change_points, n_frame_per_seg, picks

    def run(self, video_path: PathLike):
        n_frames, features, fps = self.get_features(video_path)
        print(f"--- done get feature --- n_frames:{n_frames}, len features: {len(features)}") 
        cps, nfps, picks = self.kts(n_frames, features)
        print(f"# of change points: {len(cps)}")
        print("--- done kts ---")
        return n_frames, features, cps, nfps, picks


