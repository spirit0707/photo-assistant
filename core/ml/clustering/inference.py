import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def normalize_features(features_arr):
    scaler = StandardScaler()
    normed = scaler.fit_transform(features_arr)
    return normed, scaler

def fit_kmeans(features_list, n_clusters=3, save_path=None):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features_list)
    if save_path:
        joblib.dump(kmeans, save_path)
    return kmeans

def load_kmeans(path):
    return joblib.load(path)

def predict_cluster(kmeans, features):
    return kmeans.predict([features])[0]

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