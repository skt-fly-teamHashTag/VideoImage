#!/usr/bin/env python3 -u
# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import logging
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from fairseq import distributed_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.utils import reset_logging
from omegaconf import DictConfig

from tqdm import tqdm

from utils import checkpoint_utils
from utils.eval_utils import eval_step, merge_results, eval_caption
from utils.zero_shot_utils import zero_shot_step

from tasks.mm_tasks.caption import CaptionTask
from models.ofa import OFAModel
from PIL import Image

def infer_main():

    # Register refcoco task
    tasks.register_task('caption', CaptionTask)

    # turn on cuda if GPU is available
    use_cuda = torch.cuda.is_available()
    # use fp16 only when GPU is available
    use_fp16 = False

    # Load pretrained ckpt & config
    overrides={"eval_cider":False, "beam":5, "max_len_b":16, "no_repeat_ngram_size":3, "seed":7}
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths('./checkpoints/caption_base_best.pt'),
            arg_overrides=overrides
        )

    # Move models to GPU
    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Initialize generator
    generator = task.build_generator(models, cfg.generation)

    # Image transform
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    patch_resize_transform = transforms.Compose(
        [
            lambda image: image.convert("RGB"),
            transforms.Resize((cfg.task.patch_image_size, cfg.task.patch_image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    # Text preprocess
    bos_item = torch.LongTensor([task.src_dict.bos()])
    eos_item = torch.LongTensor([task.src_dict.eos()])
    pad_idx = task.src_dict.pad()
    def encode_text(text, length=None, append_bos=False, append_eos=False):
        s = task.tgt_dict.encode_line(
            line=task.bpe.encode(text),
            add_if_not_exist=False,
            append_eos=False
        ).long()
        if length is not None:
            s = s[:length]
        if append_bos:
            s = torch.cat([bos_item, s])
        if append_eos:
            s = torch.cat([s, eos_item])
        return s

    # Construct input for caption task
    def construct_sample(image: Image):
        patch_image = patch_resize_transform(image).unsqueeze(0)
        patch_mask = torch.tensor([True])
        src_text = encode_text(" what does the image describe?", append_bos=True, append_eos=True).unsqueeze(0)
        src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
        sample = {
            "id":np.array(['42']),
            "net_input": {
                "src_tokens": src_text,
                "src_lengths": src_length,
                "patch_images": patch_image,
                "patch_masks": patch_mask
            }
        }
        return sample
      
    # Function to turn FP32 to FP16
    def apply_half(t):
        if t.dtype is torch.float32:
            return t.to(dtype=torch.half)
        return t

    ## captioning 
    f = open("./sample_data/caption.txt", 'w')

    thumb_imgs = np.load('/content/Drive/MyDrive/skt-flyai /skt-fly-teamHashTag/DSNet/output/test/testdata3_output.npy', allow_pickle=True)
    thumb_half = thumb_imgs[::7, :]

    for i in tqdm(range(len(thumb_half))):
        im = Image.fromarray(thumb_half[i][0])
        sample = construct_sample(im)
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample
        with torch.no_grad():
            result, scores = eval_caption(task, generator, models, sample)

        ## .txt 파일로 저장하기 
        f.write(result[0]['caption']+'\n')

    f.close()



if __name__ == "__main__":
    infer_main()




