import torch
import torch.nn.functional as F

from icecream import ic

class mean_sampling(object):
    def __init__(self,):
        pass
    
    @torch.no_grad()
    def __call__(self, video_features: torch.Tensor):
        frame_features = torch.mean(video_features.float(), dim=0).unsqueeze(0)
        return frame_features

class uniform_sampling(object):
    def __init__(self,n_frames):
        self.n_frames = n_frames

    @torch.no_grad()
    def __call__(self, video_features: torch.Tensor):
        index = torch.linspace(0, video_features.size(0) - 1, self.n_frames, dtype=torch.long) # Uniform Sampling
        frame_features = video_features[index].clone().float()
        return frame_features

class random_sampling(object):
    def __init__(self,n_frames):
        self.n_frames = n_frames

    @torch.no_grad()
    def __call__(self, video_features: torch.Tensor):
        index = torch.randperm(video_features.size(0))[:self.n_frames]
        index = torch.sort(index)[0]
        frame_features = video_features[index].clone().float()
        return frame_features

class cluster_sampling(object):
    def __init__(self,n_frames):
        self.n_frames = n_frames

    @torch.no_grad()
    def kmeans_cosine_density(self, input_feature, k, max_iterations=5000):
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
    def __call__(self, video_features: torch.Tensor):
        video_features = video_features.float()
        frame_features = self.kmeans_cosine_mean(video_features, self.n_frames)
        return frame_features
    
class redensity_cosin_sampling(object):
    def __init__(self, n_frames):
        self.n_frames = n_frames

    def __call__(self, video_features: torch.Tensor) -> torch.Tensor:
        video_norm = F.normalize(video_features, p=2, dim=1)  # shape: (N, D)
        cos_sim_matrix = torch.matmul(video_norm, video_norm.t())

        density = cos_sim_matrix.mean(dim=1)  # shape: (N,)

        epsilon = 1e-12
        scores = 1.0 / (density + epsilon)    # shape: (N,)

        probabilities = scores.softmax(dim=-1)
        chosen_indices = torch.multinomial(probabilities, self.n_frames, replacement=False)

        return video_features[chosen_indices]
