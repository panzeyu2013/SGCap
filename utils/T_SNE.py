import numpy as np
import torch
import os
import clip
import json
from sklearn.manifold import TSNE
from utils.sampling import mean_sampling, random_sampling, cluster_sampling, uniform_sampling, redensity_cosin_sampling
from typing import List, Tuple
import matplotlib.pyplot as plt

def draw_tsne(input: List[Tuple[str, torch.Tensor]], save_path: str, perplexity=30):    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    labels, tensor_list = zip(*input)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if os.path.exists(os.path.join(save_path, '_vs_'.join(labels)+".pdf")):
        return
    
    sizes = [tensor.size(0) for tensor in tensor_list]
    all_tensor = torch.cat(tensor_list, dim=0)
    video_tsne = tsne.fit_transform(all_tensor.numpy())
    
    split_tsne = []
    start = 0
    for size in sizes:
        end = start + size
        split_tsne.append(video_tsne[start:end])
        start = end
    
    plt.clf()
    
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
    for idx, tsne_data in enumerate(split_tsne):
        plt.scatter(tsne_data[:, 0], tsne_data[:, 1], label=labels[idx], color=colors[idx % len(colors)])

    plt.legend()
    plt.title('t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig(os.path.join(save_path, '_vs_'.join(labels)+".pdf"))


def get_text_feature(video_feature: torch.Tensor, text_bank: torch.Tensor, device="cuda:0"):
    def weighted_sum(sim: torch.Tensor, text_features: torch.Tensor):
        sim = (sim*100).softmax(dim=-1)
        text_features = sim @ text_features
        return text_features
    
    text_bank = text_bank.to(device).float()
    norm_text_features = text_bank / text_bank.norm(dim=-1,keepdim=True)
    frame_features = video_feature.to(device).float()
    norm_frame_features = frame_features / frame_features.norm(dim=-1,keepdim=True)
    frames = []
    for idx in range(video_feature.size(0)):
        norm_cur_frame = norm_frame_features[idx]
        sim = norm_cur_frame @ norm_text_features.T
        cur_frame = weighted_sum(sim, text_bank)
        frames.append(cur_frame.unsqueeze(0))
    frames = torch.cat(frames, dim=0).float().cpu()
    return frames

if __name__ == "__main__":
    idx = 0
    model, processor = clip.load("ViT-B/32")
    captions = json.load(open("../data/extract/msrvtt/annotations.json",mode='r'))['caption']
    sentences = captions["video"+str(idx)]
    with torch.no_grad():
        gt_feature = model.encode_text(clip.tokenize(sentences).to("cuda:0")).cpu()

    video = torch.load("../data/extract/msrvtt/video_feature/video"+str(idx)+".pth")
    text_bank = torch.load("../data/extract/msrvtt/all_text_features.pth")

    video_uniform = uniform_sampling(5)(video.clone())
    video_random = random_sampling(5)(video.clone())
    video_cluster = cluster_sampling(5)(video.clone())
    video_redensity_cosine = redensity_cosin_sampling(5)(video.clone())
    video_mean = mean_sampling()(video.clone())

    # all_vs_gt_vs_uniform

    all_video_text = get_text_feature(video, text_bank)
    text_uniform = get_text_feature(video_uniform, text_bank)
    text_random = get_text_feature(video_random, text_bank)
    text_cluster = get_text_feature(video_cluster, text_bank)
    text_redensity_cosine = get_text_feature(video_redensity_cosine, text_bank)
    text_mean = get_text_feature(video_mean, text_bank)

    retrieved = []
    text_bank = text_bank.float().to("cuda:0")
    norm_text_bank = text_bank / text_bank.norm(dim=-1,keepdim=True)
    for i in range(gt_feature.size(0)):
        target_gt = gt_feature[i].clone().float().to("cuda:0")
        norm_targetz_gt = target_gt / target_gt.norm(dim=-1,keepdim=True)
        
        similarity = norm_targetz_gt @ norm_text_bank.T
        score, index = torch.topk(similarity, dim=-1, k=1)
        retrieved.append(text_bank[index].clone().cpu())
    
    retrieved = torch.cat(retrieved, dim=0)
    draw_tsne([("frames", all_video_text), ("cosine", retrieved), ("gt", gt_feature)], "../figures/msrvtt_text_distribution/")

    draw_tsne([("frames", all_video_text), ("gt", gt_feature), ("uniform", text_uniform)], "../figures/msrvtt_text_distribution/"+str(idx),)
    draw_tsne([("frames", all_video_text), ("gt", gt_feature), ("random", text_random)], "../figures/msrvtt_text_distribution/"+str(idx),)
    draw_tsne([("frames", all_video_text), ("gt", gt_feature), ("cluster", text_cluster)], "../figures/msrvtt_text_distribution/"+str(idx),)
    draw_tsne([("frames", all_video_text), ("gt", gt_feature), ("mean", text_mean)], "../figures/msrvtt_text_distribution/"+str(idx),)
    draw_tsne([("frames", all_video_text), ("gt", gt_feature), ("redensity_cosine", text_redensity_cosine)], "../figures/msrvtt_text_distribution/"+str(idx),)