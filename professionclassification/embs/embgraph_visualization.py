import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

data = np.load("embs/numpy_embs.npy", allow_pickle=True)


embeddings = data 

embeddings_values = np.array([item["embedding"] for item in embeddings])
labels = np.array([item["label"] for item in embeddings])
file_paths = np.array([item["file_path"] for item in embeddings])

print(f"Загружено {len(embeddings)} эмбеддингов размерности {embeddings_values.shape[1]}")

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
)
embeddings_2d = tsne.fit_transform(embeddings_values)

plt.figure(figsize=(12, 8))

palette = sns.color_palette("husl", len(set(labels)))

for i, label in enumerate(set(labels)):
    idx = np.where(labels == label)
    plt.scatter(
        embeddings_2d[idx, 0],
        embeddings_2d[idx, 1],
        color=palette[i],
        label=label,
        alpha=0.7,
        s=50
    )

plt.legend(title="Классы", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("Визуализация эмбеддингов (t-SNE)", fontsize=14)
plt.xlabel("Dimension 1", fontsize=12)
plt.ylabel("Dimension 2", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("embs/embeddings_tsne.png", dpi=300, bbox_inches='tight')
plt.show()
print("График сохранён как 'embeddings_tsne.png'")