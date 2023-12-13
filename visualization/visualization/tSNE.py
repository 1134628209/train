# use t-SNE to show the feature distribution

from sklearn import manifold
from einops import reduce


def plt_raw_tsne(ax, data, label, per):
    data = data.cpu().detach().numpy()
    data = reduce(data, 'b n e -> b e', reduction='mean')
    label = label.cpu().detach().numpy()

    tsne = manifold.TSNE(n_components=2, perplexity=per, init='pca', random_state=166, learning_rate='auto')
    X_tsne = tsne.fit_transform(data)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    for i in range(X_norm.shape[0]):
        color = 'red' if label[i] == 1 else '#80BFFF'
        ax.scatter(X_norm[i, 0], X_norm[i, 1], color=color)
        ax.set_xticks([])
        ax.set_yticks([])


def plt_target_tsne(ax, data, label, per):
    data = data.cpu().detach().numpy()
    label = label.cpu().detach().numpy()

    tsne = manifold.TSNE(n_components=2, perplexity=per, init='pca', random_state=166, learning_rate='auto')
    X_tsne = tsne.fit_transform(data)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    for i in range(X_norm.shape[0]):
        color = 'red' if label[i] == 1 else '#80BFFF'
        ax.scatter(X_norm[i, 0], X_norm[i, 1], color=color)
        ax.set_xticks([])
        ax.set_yticks([])
