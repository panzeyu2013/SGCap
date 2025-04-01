import torch
import clip
import json
import os
import cv2
import argparse
from PIL import Image
from tqdm import tqdm

@torch.no_grad()
def extract_video_features(args, video_path, model, processor):
    vcap = cv2.VideoCapture()
    vcap.open(video_path)
    total_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    for idx in range(total_frames):
        _, frame = vcap.read()
        if frame is not None:
            tensor = processor(Image.fromarray(frame))
            frames.append(tensor.unsqueeze(0))
    frames = torch.cat(frames, dim=0)
    feature = model.encode_image(frames.to(args.device))
    
    vcap.release()
    return feature.cpu()

def main(args):
    batch_size = args.batch_size
    device = args.device

    model, processor = clip.load(args.clip_name)
    model.eval()

    captions = []
    with open(args.text_path,mode='r') as f:
        file = json.load(f)
    
    name2cap = {}

    train_list = []
    for item in file['train']:
        train_list.append(item['videoID'])
        name2cap[item['videoID']] = item['enCap']
        captions += item['enCap']

    val_list = []
    for item in file['val']:
        val_list.append(item['videoID'])
        name2cap[item['videoID']] = item['enCap']
        if args.text_split == "all":
            captions += item['enCap']
    
    test_list = []
    for item in file['test']:
        test_list.append(item['videoID'])
        name2cap[item['videoID']] = item['enCap']
        if args.text_split == "all":
            captions += item['enCap']

    video_list = train_list + val_list + test_list
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    if not os.path.exists(os.path.join(args.save_path, "annotations.json")):
        with open(os.path.join(args.save_path, "annotations.json"), mode='w') as p:
            json.dump({'caption': name2cap, 'train': train_list, 'val': val_list, 'test': test_list}, p)

    if not os.path.exists(os.path.join(args.save_path, args.text_split + "_captions.pth")):
        torch.save(captions, os.path.join(args.save_path, args.text_split + "_captions.pth"))

    if not os.path.exists(os.path.join(args.save_path, args.text_split + "_text_features.pth")):
        features = []
        with torch.no_grad():
            for iter in tqdm(range(len(captions)//batch_size + 1)):
                batched_captions = captions[iter*batch_size : (iter+1)*batch_size]
                batched_captions = clip.tokenize(batched_captions,truncate=True).to(device)
                batched_captions = model.encode_text(batched_captions).cpu()
                features.append(batched_captions)
    
        features = torch.cat(features, dim=0)
        torch.save(features,os.path.join(args.save_path, args.text_split + "_text_features.pth"))
    else:
        features = torch.load(os.path.join(args.save_path, args.text_split + "_text_features.pth"))


    if not os.path.exists(os.path.join(args.save_path, args.text_split + "_neighbors_indice.pth")):
        neighbors = []
        with torch.no_grad():
            features = features.to(device)
            features /= features.norm(dim=-1, keepdim=True)
            for idx in tqdm(range(features.size(0))):
                current_caption = features[idx].clone().unsqueeze(0).to(device)
                similarity = current_caption @ features.T
                similarity[0,idx] = 0 # exclude the current caption
                topk_value, topk_indice = torch.topk(similarity, k=20, dim=-1)
                neighbors.append(topk_indice.cpu())
        neighbors = torch.cat(neighbors,dim=0)
        torch.save(neighbors, os.path.join(args.save_path, args.text_split + "_neighbors_indice.pth"))
    else:
        neighbors = torch.load(os.path.join(args.save_path, args.text_split + "_neighbors_indice.pth"))

    if not os.path.exists(os.path.join(args.save_path, args.text_split + "_all_cosine.pth")):
        neighbors = []
        with torch.no_grad():
            features = features.to(device)
            features /= features.norm(dim=-1, keepdim=True)
            for idx in tqdm(range(features.size(0))):
                current_caption = features[idx].clone().unsqueeze(0).to(device)
                similarity = current_caption @ features.T
                similarity[0,idx] = 0 # exclude the current caption
                similarity = torch.where(similarity>=1, 0, similarity)
                topk_value, topk_indice = torch.topk(similarity, k=20, dim=-1)
                current = [(int(topk_indice[0][i].cpu()), float(topk_value[0][i].cpu())) for i in range(20)]
                neighbors.append(current)
        torch.save(neighbors, os.path.join(args.save_path, args.text_split + "_all_cosine.pth"))
    else:
        neighbors = torch.load(os.path.join(args.save_path, args.text_split + "_all_cosine.pth"))


    fail_list = []
    video_root = os.path.join(args.save_path, "video_feature")
    if not os.path.exists(video_root):
        os.makedirs(video_root)
    for video_id in tqdm(video_list):
        feature_path = os.path.join(video_root, video_id + ".pth")
        if not os.path.exists(feature_path):
            video_path = os.path.join(args.video_path, video_id + ".mp4")
            if os.path.exists(video_path):
                video_features = extract_video_features(args, video_path, model, processor)
                torch.save(video_features, feature_path)
    torch.save(fail_list, os.path.join(args.save_path, "fail.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_path",default="../data/origin/VATEX/annotation/train_val_test.json",type=str)
    parser.add_argument("--video_path",default="../data/origin/VATEX/VATEX-videos",type=str)
    parser.add_argument("--save_path",default="../data/extract/vatex")
    parser.add_argument("--text_split",default="all",type=str)
    parser.add_argument("--clip_name",default="ViT-B/32",type=str)
    parser.add_argument("--device",default="cuda:0",type=str)
    parser.add_argument("--batch_size",default=512,type=int)

    args = parser.parse_args()
    main(args)