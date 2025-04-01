import torch
import torch.nn as nn
from argparse import ArgumentParser
from tqdm import tqdm
from typing import List
from torch.utils.data import Dataset, DataLoader
from nltk import pos_tag, word_tokenize

class c_dataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.captions = torch.load(path)
        # self.ngrams = [set(self._extract_ngram(ngram)) for ngram in self.captions]
        self.nouns_verbs = [set(self._extract_nouns_verbs(sentence)) for sentence in tqdm(self.captions)]

    def _extract_nouns_verbs(self, text: str) -> List[str]:
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        return [word for word, tag in tagged if tag in ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']]

    def _extract_ngram(self, words:str, n:int = 2) -> List[str]:
        ngrams = zip(*[words[i:] for i in range(n)])
        return [" ".join(ngram) for ngram in ngrams]

    def jaccard_similarity(self, set1, set2):
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union != 0 else 0

    def __getitem__(self, index):
        sims = {}
        for j in range(len(self.captions)):
            if index == j:
                continue
            sims[j] = self.jaccard_similarity(self.nouns_verbs[index],self.nouns_verbs[j])

        captions = set()
        output = []
        for k, v in sims.items():
            if self.captions[k] not in captions and v < 1:
                output.append((k,v))
                captions.add(self.captions[k])

        output.sort(key=lambda x:x[1],reverse=True)
        output = output[:20]
        return output

    def __len__(self,):
        return len(self.captions)
    
class collate_fn(object):
    def __call__(self, batch):
        return batch[0]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../data/extract/msrvtt/all_captions.pth")
    parser.add_argument("--output_path", type=str, default="../data/extract/msrvtt/all_nv_jaccard.pth")
    parser.add_argument("--num_workers", type=int, default=16)

    args = parser.parse_args()

    dataloader = DataLoader(dataset=c_dataset(args.data_path),
                        collate_fn=collate_fn(),
                        shuffle=False,
                        num_workers=args.num_workers,
                        batch_size=1)
    output = []
    for sims in tqdm(dataloader):
    # print(index, sims)
        output.append(sims)
    
    torch.save(output, args.output_path)