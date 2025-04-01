import argparse
import itertools
import json
import re
import os
import random
import time
import torch
from functools import partial
from typing import Sequence, Dict, Any
import torch.distributed
from peft import PeftModel, PeftConfig
from tqdm import tqdm
from PIL import Image
from transformers import AutoModel, AutoConfig, AutoTokenizer, PreTrainedTokenizer
from model.configuration_model import GCapconfig
from model.build import build_tokenizer
from model.modeling import GCap
from model.sample import (one_frame_sampling, 
                          mean_sampling, 
                          uniform_sampling, 
                          random_sampling, 
                          cluster_sampling,
                          redensity_sampling)
from eval.eval import language_eval
from icecream import ic

sample_strategy = {
    'one': one_frame_sampling,
    'mean': mean_sampling,
    'uniform': uniform_sampling,
    'random': random_sampling,
    'cluster': cluster_sampling,
    'redensity': redensity_sampling
}

class CaptionDataset(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer =  tokenizer

        self.video_feature_folder = os.path.join(config.data_root, config.dataset_name, "video_feature")
        with open(os.path.join(config.data_root, config.dataset_name, "annotations.json"),mode='r') as p:
            file = json.load(p)
            name2cap = file['caption']
            name_list = file[config.video_split]

        self.name2cap = name2cap

        self.name_list = []
        for name in name_list:
            video_path = os.path.join(self.video_feature_folder, name + ".pth") # in case video not exist
            if os.path.exists(video_path):
                self.name_list.append(name)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        name = self.name_list[index]
        captions = self.name2cap[name]
        video_feature = torch.load(os.path.join(self.video_feature_folder, name + ".pth"))

        return {
            'video_feature': video_feature.float(),
            'captions': captions
        }


class collate_fn(object):
    def __init__(self,) -> None:
        pass
    
    def __call__(self, instances: Sequence[Dict]) -> list[str, torch.Tensor | Any]:
        video_features = [_['video_feature'] for _ in instances]
        captions = [_['captions'] for _ in instances]

        return  video_features, captions


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/checkpoint')
    parser.add_argument('--lora_checkpoint', type=str, default='')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str, default='msrvtt')
    parser.add_argument('--caption_from', type=str, default='msrvtt')
    parser.add_argument('--caption_split', type=str, default='all')
    parser.add_argument('--video_split', type=str, default='test')
    parser.add_argument('--sampling', type=str, default='uniform')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--result_file',type=str, default='./checkpoints/test.json')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--device', type=str, default="cuda:0")
    config = parser.parse_args()

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))
    # os.environ['CUDA_VISIBLE_DEVICES'] = os.getenv('LOCAL_RANK', "0")

    tokenizer = build_tokenizer(config.checkpoint)
    model_config = GCapconfig.from_pretrained(config.checkpoint)
    model = GCap.from_pretrained(pretrained_model_name_or_path=config.checkpoint,config=model_config)
    if os.path.exists(config.lora_checkpoint):
        lora_config = PeftConfig.from_pretrained(config.lora_checkpoint)
        model = PeftModel.from_pretrained(model, config.lora_checkpoint, config=lora_config)
    model.float().cuda()

    random.seed(config.seed)

    dataset = CaptionDataset(config, tokenizer)
    collate = collate_fn()

    test_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate,
    )

    text_feature_bank_path = os.path.join(config.data_root, config.caption_from, config.caption_split + "_text_features.pth")
    text_features_bank = torch.load(text_feature_bank_path)

    text_features_bank = text_features_bank.float().to(config.device)
    norm_text_feature_bank = text_features_bank / text_features_bank.norm(keepdim=True, dim=-1)

    sample_fn = sample_strategy[config.sampling](config)

    gt_captions = []
    pred_caption = []

    model.eval()
    for (video_features, captions) in tqdm((test_loader)):

        batch_size = len(video_features)
        
        args = {
            'do_sample': False,
            'num_beams': 5,
            'max_new_tokens': 60,
            'min_new_tokens': 8,
            'length_penalty': 0,
            'num_return_sequences': 1,
            'use_cache': True,
        }

        text_features = sample_fn(video_features, text_features_bank)

        video_means = []
        for idx in range(len(video_features)):
            cur_video = video_features[idx].to(config.device)
            cur_video = torch.mean(cur_video,dim=0)
            norm_cur_video = cur_video / cur_video.norm(dim=-1,keepdim=True)
            sim = norm_cur_video @ norm_text_feature_bank.T
            sim = (sim*100).softmax(dim=-1)
            embedding = sim @ text_features_bank
            video_means.append(embedding.unsqueeze(0))
        video_means = torch.cat(video_means,dim=0)

        output = model.generate(
            visual_feature=video_means.to(model.dtype),
            text_features=text_features.to(model.dtype),
            attention_mask = None,
            **args
        )

        pred_caption += tokenizer.batch_decode(output[0].cpu(),skip_special_tokens=True)
        gt_captions += captions
        
    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_gt_captions = [None for _ in range(world_size)]
    merged_pred_captions = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_gt_captions, gt_captions)
    torch.distributed.all_gather_object(merged_pred_captions, pred_caption)

    merged_gt_captions = [_ for _ in itertools.chain.from_iterable(merged_gt_captions)] #from List[List[Any]] -> List[Any]
    merged_pred_captions = [
        _ for _ in itertools.chain.from_iterable(merged_pred_captions)
    ]


    if torch.distributed.get_rank() == 0:
        metrics = language_eval(merged_pred_captions,merged_gt_captions)
        
        json_list = []
        for i in range(len(merged_gt_captions)):
            json_list.append({
                'idx':i,
                'pd_caption': merged_pred_captions[i],
                'gt_caption': merged_gt_captions[i],
            })
        
        metrics.update({'result': json_list})

        json.dump(metrics, open(config.result_file, mode='w'))        
    torch.distributed.barrier()