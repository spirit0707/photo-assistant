import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_clusters(features, labels, save_path=None):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='viridis', s=60)
    plt.title('Кластеры изображений (PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar(scatter, label='Кластер')
    if save_path:
        plt.savefig(save_path)
    plt.close()
    return save_path

__all__ = ["visualize_clusters"] 