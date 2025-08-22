import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class EmbeddingVisualizer:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def plot_2d(self, labels=None, method="pca"):
        if method == "pca":
            reducer = PCA(n_components=2)
        else:
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)

        reduced = reducer.fit_transform(self.embeddings)

        plt.figure(figsize=(8, 6))
        plt.scatter(reduced[:, 0], reduced[:, 1], c="blue", alpha=0.6)

        if labels is not None:
            for i, label in enumerate(labels):
                plt.annotate(label, (reduced[i, 0], reduced[i, 1]))

        plt.title(f"2D Visualization ({method.upper()})")
        plt.show()

    def plot_3d(self, labels=None, method="pca"):
        if method == "pca":
            reducer = PCA(n_components=3)
        else:
            reducer = TSNE(n_components=3, random_state=42, perplexity=30)

        reduced = reducer.fit_transform(self.embeddings)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], c="red", alpha=0.6)

        if labels is not None:
            for i, label in enumerate(labels):
                ax.text(reduced[i, 0], reduced[i, 1], reduced[i, 2], label)

        ax.set_title(f"3D Visualization ({method.upper()})")
        plt.show()