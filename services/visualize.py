import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class EmbeddingVisualizer:
    def __init__(self, embeddings):
        # Convert to numpy array if it's a list
        if isinstance(embeddings, list):
            self.embeddings = np.array(embeddings)
        else:
            self.embeddings = embeddings

    def plot_2d(self, labels=None, method="pca"):
        if method == "pca":
            reducer = PCA(n_components=2)
        else:
            # Adjust perplexity based on number of samples
            n_samples = self.embeddings.shape[0]
            perplexity = min(30, max(1, n_samples - 1))
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)

        reduced = reducer.fit_transform(self.embeddings)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(reduced[:, 0], reduced[:, 1], c="blue", alpha=0.6)

        if labels is not None:
            for i, label in enumerate(labels):
                ax.annotate(label, (reduced[i, 0], reduced[i, 1]), fontsize=8)

        ax.set_title(f"2D Visualization ({method.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        plt.tight_layout()
        return fig

    def plot_3d(self, labels=None, method="pca"):
        if method == "pca":
            reducer = PCA(n_components=3)
        else:
            # Adjust perplexity based on number of samples
            n_samples = self.embeddings.shape[0]
            perplexity = min(30, max(1, n_samples - 1))
            reducer = TSNE(n_components=3, random_state=42, perplexity=perplexity)

        reduced = reducer.fit_transform(self.embeddings)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], c="red", alpha=0.6)

        if labels is not None:
            for i, label in enumerate(labels):
                ax.text(reduced[i, 0], reduced[i, 1], reduced[i, 2], label, fontsize=8)

        ax.set_title(f"3D Visualization ({method.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
        plt.tight_layout()
        return fig
