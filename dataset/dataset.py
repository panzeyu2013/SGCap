import torch
import os
import re
import argparse
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import Dict, Optional, Sequence, List, Any, Union

from icecream import ic
from tqdm import tqdm

class text_dataset(Dataset):
    def __init__(self, args, tokenizer, split, media_tokens=['<feature>']):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.max_length = args.max_length

        self.media_tokens = {k: -int(i+1) for i, k in enumerate(media_tokens)}
        self.media_lengths = {'<feature>': args.k_neighbors}

        caption_path = os.path.join(args.data_root, 
                                    args.dataset_name, 
                                    args.data_split + "_captions.pth")
        neighbor1_path = os.path.join(args.data_root,
                                     args.dataset_name,
                                     args.data_split + "_all_cosine.pth")
        neighbor2_path = os.path.join(args.data_root,
                                     args.dataset_name,
                                     args.data_split + "_nv_jaccard.pth")
        feature_path = os.path.join(args.data_root,
                                    args.dataset_name,
                                    args.data_split + "_text_features.pth")
        
        self.captions = torch.load(caption_path)
        self.neighbors1 = torch.load(neighbor1_path)
        self.neighbors2 = torch.load(neighbor2_path)
        
        self.features_bank: torch.Tensor = torch.load(feature_path)
        self.features_bank = self.features_bank.float()

        self.text_feat_std = torch.std(self.features_bank, dim=0)
        # self.text_feat_std = torch.mean(torch.std(self.features_bank, dim=0))

        num_samples = int(len(self.captions)*0.95*self.args.percent)
        if split == "train":
            self.features = self.features_bank[:num_samples]
            self.neighbors1 = self.neighbors1[:num_samples]
            self.neighbors2 = self.neighbors2[:num_samples]
            
        elif split == "val":
            self.features = self.features_bank[num_samples:]
            self.neighbors1 = self.neighbors1[num_samples:]
            self.neighbors2 = self.neighbors2[num_samples:]

    def __len__(self):
        return self.features.size(0)
    
    def __getitem__(self, index):
        text_feature = self.features[index].clone()
        noise = torch.normal(mean = torch.zeros_like(self.text_feat_std), std = self.text_feat_std)
        # noise = torch.randn(512) * self.text_feat_std
        text_feature = text_feature + self.args.noise * noise

        idx2value = {}
        for i in range(len(self.neighbors1[index])):
            idx, value = self.neighbors1[index][i]
            if not idx in idx2value.keys():
                idx2value[idx] = value * self.args.sigma
            else:
                idx2value[idx] += value * self.args.sigma
        for i in range(len(self.neighbors2[index])):
            idx, value = self.neighbors2[index][i]
            if not idx in idx2value.keys():
                idx2value[idx] = value * (1-self.args.sigma)
            else:
                idx2value[idx] += value * (1-self.args.sigma)
        
        idx, value = [k for k,v in idx2value.items()] , [v for k, v in idx2value.items()]

        idx = torch.tensor(idx, dtype=torch.long)
        prob = torch.tensor(value, dtype=torch.float)
        # prob = torch.softmax(value, dim=-1)

        selected_prob, selected = torch.topk(prob, self.args.k_neighbors,sorted=False)
        # selected = torch.multinomial(prob, self.args.k_neighbors)
        selected_index = idx[selected]

        text_neighbors = self.features_bank[selected_index, :].clone()
        shape = torch.zeros_like(self.text_feat_std).unsqueeze(0).expand((self.args.k_neighbors,-1))
        noise = torch.normal(mean = shape, std = shape + self.text_feat_std)
        # noise = torch.randn((self.args.k_neighbors, 512)) * self.text_feat_std
        text_neighbors = text_neighbors + self.args.neighbor_noise * noise

        all_index = torch.cat([torch.tensor([index]), selected_index],dim=0)
        all_prob = torch.cat([torch.ones([1]) * self.args.p, selected_prob],dim=0).softmax(dim=-1)

        caption_prob_selected = torch.multinomial(all_prob, 1)
        caption_index_selected = all_index[caption_prob_selected]

        caption = self.captions[caption_index_selected]
        # caption = self.captions[index]
        caption = '<feature>:' + caption
        text = self._extract_tokens_from_text(caption, self.max_length)

        input_ids = text['input_ids']
        label_mask = text['label_mask']
        non_padding_mask = text['non_padding_mask']
        non_media_mask = text['non_media_mask']

        return {
            'text_feature': text_feature,
            'text_neighbors': text_neighbors,
            'input_ids': input_ids,
            'label_mask': label_mask,
            'non_padding_mask': non_padding_mask,
            'non_media_mask': non_media_mask,
        }
    
    def _extract_tokens_from_text(self, data, max_length):
        enc_chunk = []
        enc_length = 0
        if self.tokenizer.bos_token_id > 0:
            prompt_chunk = [self.tokenizer.bos_token_id]
        else:
            prompt_chunk = []

        pattern = '|'.join(map(re.escape, list(self.media_tokens.keys())))
        chunk_strs = re.split(f'({pattern})', data)
        chunk_strs = [x for x in chunk_strs if len(x) > 0]
        for idx, chunk_str in enumerate(chunk_strs):
            if enc_length > max_length + 1:
                break

            if idx == 0:
                if chunk_str in self.media_tokens:
                    if enc_length + self.media_lengths[chunk_str] > max_length + 1:
                        break
                    else:
                        enc_chunk = [self.media_tokens[chunk_str]] * \
                        self.media_lengths[chunk_str]
                        enc_length = len(enc_chunk)
                        label_chunk = [0] * self.media_lengths[chunk_str]
                else:
                    enc_chunk = prompt_chunk + \
                    self.tokenizer(chunk_str, add_special_tokens=False)['input_ids']
                    enc_length = len(enc_chunk)
                    label_chunk = [1] * enc_length
            elif chunk_strs[idx-1] in self.media_tokens:
                if chunk_str in self.media_tokens:
                    if enc_length + self.media_lengths[chunk_str] > max_length + 1:
                        break
                    else:
                        enc_chunk += [self.media_tokens[chunk_str]] * \
                        self.media_lengths[chunk_str]
                        enc_length += self.media_lengths[chunk_str]
                        label_chunk += [0] * self.media_lengths[chunk_str]
                else:
                    curr_chunk = prompt_chunk + self.tokenizer(chunk_str, add_special_tokens=False)['input_ids']
                    curr_length = len(curr_chunk)
                    if enc_length + curr_length > max_length:
                        curr_chunk = curr_chunk[:max_length-enc_length]
                    curr_chunk += [self.tokenizer.eos_token_id]
                    curr_length = len(curr_chunk)
                    enc_chunk += curr_chunk
                    enc_length += curr_length
                    label_chunk += [0] + [1] * (curr_length - 1)
            else:
                raise NotImplemented

        if enc_length < max_length + 1:
            padding_chunk = [self.tokenizer.pad_token_id] * \
            (max_length + 1 - enc_length)
            label_chunk += [0] * (max_length + 1 - enc_length)
            enc_chunk += padding_chunk

        non_padding_mask = [1 if i < enc_length - 1 else 0 for i in range(max_length)]
        
        enc_chunk = torch.tensor(enc_chunk, dtype=torch.long)
        label_chunk = torch.tensor(label_chunk, dtype=torch.long)
        non_padding_mask = torch.tensor(non_padding_mask, dtype=torch.long)

        tmp_enc_chunk = enc_chunk.clone()
        tmp_enc_chunk[tmp_enc_chunk >= 0] = 1
        tmp_enc_chunk[tmp_enc_chunk < 0] = 0
        non_media_mask = tmp_enc_chunk.clone().long()
        non_media_mask = non_media_mask[1:]

        return {
            'input_ids': enc_chunk,
            'label_mask': label_chunk,
            'non_padding_mask': non_padding_mask,
            'non_media_mask': non_media_mask,
        }


class text_collate(object):
    def __init__(self):
        pass

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor | Any]:
        text_feature = torch.cat([_['text_feature'].unsqueeze(0) for _ in instances],dim=0)
        text_neighbors = torch.cat([_['text_neighbors'].unsqueeze(0) for _ in instances], dim=0)
        input_ids = torch.cat([_['input_ids'].unsqueeze(0) for _ in instances], dim=0)
        label_mask = torch.cat([_['label_mask'].unsqueeze(0) for _ in instances], dim=0)
        non_padding_mask = torch.cat([_['non_padding_mask'].unsqueeze(0) for _ in instances], dim=0)
        non_media_mask = torch.cat([_['non_media_mask'].unsqueeze(0) for _ in instances], dim=0)

        return {
            'text_feature': text_feature,
            'text_neighbors': text_neighbors,
            'input_ids': input_ids,
            'label_mask': label_mask,
            'non_padding_mask': non_padding_mask,
            'non_media_mask': non_media_mask,
        }

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="../data")
    parser.add_argument("--dataset_name", type=str, default="msrvtt")
    parser.add_argument("--data_split", type=str, default="train")
    parser.add_argument("--k_neighbors", type=int, default=5)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    pass
    # args = get_args()
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # tokenizer.bos_token_id = 0
    # tokenizer.pad_token_id = tokenizer.eos_token_id
   
    # dataset = text_dataset(args, tokenizer, 20)
    # for item in dataset:
    #     # print(item)
    #     break