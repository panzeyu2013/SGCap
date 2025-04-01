import torch
import torch.nn.functional as F

from icecream import ic

def weighted_sum(sim: torch.Tensor, text_features: torch.Tensor):
    sim = (sim*100).softmax(dim=-1)
    text_features = sim @ text_features
    return text_features

def topk(sim: torch.Tensor, text_features: torch.Tensor, topk):
    topk_value, topk_indice = torch.topk(sim, topk, dim=-1)
    topk_indice = topk_indice.squeeze()
    text_features = text_features[topk_indice].clone()
    return text_features

class one_frame_sampling(object):
    def __init__(self, config):
        self.config = config
        self.topk = config.topk

    @torch.no_grad()
    def __call__(self, video_features, text_features):
        text_features = text_features.to(self.config.device)
        norm_text_features = text_features / text_features.norm(dim=-1,keepdim=True)
        batch_size = len(video_features)
        batch_video = []
        for batch in range(batch_size):
            cur_frame = video_features[batch][0].clone().to(self.config.device)
            cur_frame /= cur_frame.norm(dim=-1,keepdim=True)
            sim = cur_frame @ norm_text_features.T
            curr_text_features = topk(sim, text_features, self.topk)
            batch_video.append(curr_text_features.unsqueeze(0))
        batch_video = torch.cat(batch_video, dim=0)
        return batch_video

class mean_sampling(object):
    def __init__(self, config):
        self.config = config
        self.topk = config.topk

    @torch.no_grad()
    def __call__(self, video_features, text_features):
        text_features = text_features.to(self.config.device)
        norm_text_features = text_features / text_features.norm(dim=-1,keepdim=True)
        batch_size = len(video_features)
        batch_video = []
        for batch in range(batch_size):
            cur_frame = torch.mean(video_features[batch].clone(), dim=0).to(self.config.device)
            cur_frame /= cur_frame.norm(dim=-1,keepdim=True)
            sim = cur_frame @ norm_text_features.T
            curr_text_features = topk(sim, text_features, self.topk)
            batch_video.append(curr_text_features.unsqueeze(0))
        batch_video = torch.cat(batch_video, dim=0)
        return batch_video

class uniform_sampling(object):
    def __init__(self, config):
        self.config = config
        self.topk = config.topk

    @torch.no_grad()
    def __call__(self, video_features, text_features):
        text_features = text_features.to(self.config.device)
        norm_text_features = text_features / text_features.norm(dim=-1,keepdim=True)
        batch_size = len(video_features)
        batch_video = []
        for batch in range(batch_size):
            curr_video = []
            index = torch.linspace(0, video_features[batch].size(0) - 1, self.topk, dtype=torch.long) # uniform sampling
            choosen_video_feature = video_features[batch][index].clone().to(self.config.device)
            choosen_video_feature /= choosen_video_feature.norm(dim=-1, keepdim=True)
            for idx in range(self.topk):
                cur_frame = choosen_video_feature[idx]
                sim = cur_frame @ norm_text_features.T
                curr_text_features = weighted_sum(sim, text_features).unsqueeze(0)
                # curr_text_features = topk(sim, text_features, 1)
                curr_video.append(curr_text_features)
            curr_video = torch.cat(curr_video,dim=0).unsqueeze(0)
            batch_video.append(curr_video)
        batch_video = torch.cat(batch_video,dim=0)
        return batch_video


class random_sampling(object):
    def __init__(self, config):
        self.config = config
        self.topk = config.topk

    @torch.no_grad()
    def __call__(self, video_features, text_features):
        text_features = text_features.to(self.config.device)
        norm_text_features = text_features / text_features.norm(dim=-1,keepdim=True)
        batch_size = len(video_features)
        batch_video = []
        for batch in range(batch_size):
            curr_video = []
            index = torch.randperm(video_features[batch].size(0))[:self.topk]
            index = torch.sort(index)[0]
            choosen_video_feature = video_features[batch][index].clone().to(self.config.device)
            choosen_video_feature /= choosen_video_feature.norm(dim=-1, keepdim=True)
            for idx in range(self.topk):
                cur_frame = choosen_video_feature[idx]
                sim = cur_frame @ norm_text_features.T
                curr_text_features = weighted_sum(sim, text_features).unsqueeze(0)
                # curr_text_features = topk(sim, text_features, 1)
                curr_video.append(curr_text_features)
            curr_video = torch.cat(curr_video,dim=0).unsqueeze(0)
            batch_video.append(curr_video)
        batch_video = torch.cat(batch_video,dim=0)
        return batch_video

class cluster_sampling(object):
    def __init__(self, config):
        self.config = config
        self.topk = config.topk

    @torch.no_grad()
    def kmeans_cosine_density(self, input_feature, k, max_iterations=1000):
        batch_size = input_feature.size(0)

        indices = torch.randperm(batch_size)[:k]
        centroids = input_feature[indices]

        for _ in range(max_iterations):
            cosine_similarities = F.cosine_similarity(input_feature.unsqueeze(1), centroids.unsqueeze(0), dim=2)
            cosine_distances = 1 - cosine_similarities

            labels = torch.argmin(cosine_distances, dim=1)

            new_centroids = []
            for i in range(k):
                cluster_points = input_feature[labels == i].clone()
            
                if len(cluster_points) == 0:
                    new_centroids.append(input_feature[torch.randint(0, batch_size, (1,))].squeeze(0))
                else:
                    density_scores = []
                    for j in range(len(cluster_points)):
                        point = cluster_points[j]
                        similarities = F.cosine_similarity(point.unsqueeze(0), cluster_points, dim=1)
                        density_score = similarities.sum()
                        density_scores.append(density_score)
                
                    max_density_idx = torch.argmax(torch.tensor(density_scores))
                    new_centroids.append(cluster_points[max_density_idx])
        
            new_centroids = torch.stack(new_centroids)

            if torch.norm(new_centroids - centroids).max() < 1e-6:
                break

            centroids = new_centroids

        return centroids

    def kmeans_cosine_mean(self, input_feature, k, max_iterations=100):
        batch_size = input_feature.size(0)

        indices = torch.randperm(batch_size)[:k]
        centroids = input_feature[indices]

        for _ in range(max_iterations):
            cosine_similarities = F.cosine_similarity(input_feature.unsqueeze(1), centroids.unsqueeze(0), dim=2)
            cosine_distances = 1 - cosine_similarities

            labels = torch.argmin(cosine_distances, dim=1)

            new_centroids = []
            for i in range(k):
                cluster_points = input_feature[labels == i].clone()
                if len(cluster_points) > 0:
                    centroid = torch.mean(cluster_points, dim=0)
                    new_centroids.append(centroid)
                else:
                    new_centroids.append(input_feature[torch.randint(0, batch_size, (1,))].squeeze(0))  # 使用 .squeeze(0) 保证是 1D 张量
            new_centroids = torch.stack(new_centroids)

            if torch.allclose(new_centroids, centroids):
                break

            centroids = new_centroids
        return centroids

    @torch.no_grad()
    def __call__(self, video_features, text_features):
        text_features = text_features.to(self.config.device)
        norm_text_features = text_features / text_features.norm(dim=-1,keepdim=True)
        batch_size = len(video_features)
        batch_video = []
        for batch in range(batch_size):
            curr_video = []
            curr_feature = video_features[batch].clone().to(self.config.device)
            centroids = self.kmeans_cosine_density(curr_feature, self.topk)
            for idx in range(self.topk):
                cur_frame = centroids[idx]  # 使用聚类中心作为当前特征
                sim = cur_frame @ norm_text_features.T
                curr_text_features = weighted_sum(sim, text_features).unsqueeze(0)
                # curr_text_features = topk(sim, text_features, 1)
                curr_video.append(curr_text_features)
            curr_video = torch.cat(curr_video, dim=0).unsqueeze(0)
            batch_video.append(curr_video)
        batch_video = torch.cat(batch_video, dim=0)
        return batch_video
    
class redensity_sampling(object):
    def __init__(self,config):
        self.config = config
        self.topk = config.topk
    
    @torch.no_grad()
    def redensity_cosine(self, video_features: torch.Tensor) -> torch.Tensor:
        video_norm = F.normalize(video_features, p=2, dim=1)  # shape: (N, D)


        cos_sim_matrix = torch.matmul(video_norm, video_norm.t())
        density = cos_sim_matrix.mean(dim=1)  # shape: (N,)

        epsilon = 1e-12
        scores = 1.0 / (density + epsilon)    # shape: (N,)

        probabilities = scores.softmax(dim=-1)
        chosen_indices = torch.multinomial(probabilities, self.topk, replacement=False)

        return video_features[chosen_indices]


    @torch.no_grad()
    def __call__(self, video_features, text_features):
        text_features = text_features.to(self.config.device)
        norm_text_features = text_features / text_features.norm(dim=-1,keepdim=True)
        batch_size = len(video_features)
        batch_video = []
        for batch in range(batch_size):
            curr_video = []
            curr_feature = video_features[batch].clone().to(self.config.device)
            centroids = self.redensity_cosine(curr_feature)
            for idx in range(self.topk):
                cur_frame = centroids[idx]
                sim = cur_frame @ norm_text_features.T
                curr_text_features = weighted_sum(sim, text_features).unsqueeze(0)
                # curr_text_features = topk(sim, text_features, 1)
                curr_video.append(curr_text_features)
            curr_video = torch.cat(curr_video, dim=0).unsqueeze(0)
            batch_video.append(curr_video)
        batch_video = torch.cat(batch_video, dim=0)
        return batch_video