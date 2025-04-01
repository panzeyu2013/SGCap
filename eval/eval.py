from collections import OrderedDict, defaultdict

from .pycocoevalcap.bleu.bleu import Bleu
from .pycocoevalcap.cider.cider import Cider
from .pycocoevalcap.meteor.meteor import Meteor
from .pycocoevalcap.rouge.rouge import Rouge
from .pycocoevalcap.spice.spice import Spice

def text_only_language_eval(sample_seqs, groundtruth_seqs):
    assert len(sample_seqs) == len(groundtruth_seqs), 'length of sampled seqs is different from that of groundtruth seqs!'

    references, predictions = OrderedDict(), OrderedDict()
    for i in range(len(groundtruth_seqs)):
        references[i] = [groundtruth_seqs[i]]
    for i in range(len(sample_seqs)):
        predictions[i] = [sample_seqs[i]]
    
    predictions = {i: predictions[i] for i in range(len(sample_seqs))}
    references = {i: references[i] for i in range(len(groundtruth_seqs))}

    avg_bleu_score, bleu_score = Bleu(4).compute_score(references, predictions)
    print('avg_bleu_score == ', avg_bleu_score)
    avg_cider_score, cider_score = Cider().compute_score(references, predictions)
    print('avg_cider_score == ', avg_cider_score)
    avg_meteor_score, meteor_score = Meteor().compute_score(references, predictions)
    print('avg_meteor_score == ', avg_meteor_score)
    avg_rouge_score, rouge_score = Rouge().compute_score(references, predictions)
    print('avg_rouge_score == ', avg_rouge_score)
    avg_spice_score, spice_score = Spice().compute_score(references, predictions)
    print('avg_spice_score == ', avg_spice_score)

    return {'BLEU/B1': round(avg_bleu_score[0] * 100, 2), 
            'BLEU/B2': round(avg_bleu_score[1] * 100, 2), 
            'BLEU/B3': round(avg_bleu_score[2] * 100, 2), 
            'BLEU/B4': round(avg_bleu_score[3] * 100, 2),
            'CIDEr': round(avg_cider_score * 100, 2),
            'METEOR': round(avg_meteor_score * 100, 2), 
            'ROUGE': round(avg_rouge_score * 100, 2),
            'Spice': round(avg_spice_score * 100, 2)
        }



def language_eval(sample_seqs, groundtruth_seqs):
    assert len(sample_seqs) == len(groundtruth_seqs), 'length of sampled seqs is different from that of groundtruth seqs!'

    references, predictions = OrderedDict(), OrderedDict()
    for i in range(len(groundtruth_seqs)):
        references[i] = [groundtruth_seqs[i][j] for j in range(len(groundtruth_seqs[i]))]
    for i in range(len(sample_seqs)):
        predictions[i] = [sample_seqs[i]]

    predictions = {i: predictions[i] for i in range(len(sample_seqs))}
    references = {i: references[i] for i in range(len(groundtruth_seqs))}

    avg_bleu_score, bleu_score = Bleu(4).compute_score(references, predictions)
    print('avg_bleu_score == ', avg_bleu_score)
    avg_cider_score, cider_score = Cider().compute_score(references, predictions)
    print('avg_cider_score == ', avg_cider_score)
    avg_meteor_score, meteor_score = Meteor().compute_score(references, predictions)
    print('avg_meteor_score == ', avg_meteor_score)
    avg_rouge_score, rouge_score = Rouge().compute_score(references, predictions)
    print('avg_rouge_score == ', avg_rouge_score)
    avg_spice_score, spice_score = Spice().compute_score(references, predictions)
    print('avg_spice_score == ', avg_spice_score)

    return {'BLEU/B1': round(avg_bleu_score[0] * 100, 2), 
            'BLEU/B2': round(avg_bleu_score[1] * 100, 2), 
            'BLEU/B3': round(avg_bleu_score[2] * 100, 2), 
            'BLEU/B4': round(avg_bleu_score[3] * 100, 2),
            'CIDEr': round(avg_cider_score * 100, 2),
            'METEOR': round(avg_meteor_score * 100, 2), 
            'ROUGE': round(avg_rouge_score * 100, 2),
            'Spice': round(avg_spice_score * 100, 2)
        }
