import os
import pickle
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import itertools
import numpy as np
import torch
from dgl.dataloading import GraphDataLoader
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

from SpiderDatasets.MeshGraphForTrainingDataset import MeshGraphDatasetForNNTraining
from SpiderPatch.Networks import MeshNetwork


def plot_confusion_matrix(cm, target_names=None, cmap=None):
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.2f}".format(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    plt.close()
    plt.clf()


def save_confusion_matrix(cm, path, target_names=None, cmap=None):
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.2f}".format(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    plt.clf()


def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu().detach().numpy())
            max_grads.append(p.grad.abs().max().cpu().detach().numpy())
            if ave_grads[-1] == 0 or max_grads[-1] == 0:
                print(n)
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    layers = [layers[i].replace(".weight", "") for i in range(len(layers))]
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation=90)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=np.max(ave_grads) / 2)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    plt.show()


def plot_training_statistics(filename, epochs, losses, val_accuracies, val_losses, title="Training Statistics"):
    plt.plot(epochs, losses, 'sienna', label='Training Loss')
    plt.plot(epochs, val_accuracies, 'steelblue', label='Validation Accuracy')
    plt.plot(epochs, val_losses, 'cyan', label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename)
    plt.close()


def save_embeddings_statistics(path, embeddings, id):
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/patch_embedding_Epoch{id}.pkl", "wb") as embeddings_file:
        pickle.dump(embeddings, embeddings_file, protocol=-1)


def load_embeddings_statistics(path, id):
    with open(f"{path}/patch_embedding_Epoch{id}.pkl", "rb") as embeddings_file:
        return pickle.load(embeddings_file)


def plot_embeddings_statistics(paths, id, title="", labels=None):
    import matplotlib.colors as mcolors

    embeddings = []
    if labels is None:
        labels = []
        for path in paths:
            embeddings.append(load_embeddings_statistics(path, id))
            labels.append(path.split("/")[-2])
    else:
        for path in paths:
            embeddings.append(load_embeddings_statistics(path, id))

    embeddings = np.array(embeddings)

    asort = np.argsort(embeddings, axis=0)
    sort = np.sort(embeddings, axis=0)

    plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = plt.subplot(111)
    labels.reverse()
    cmap = matplotlib.colors.ListedColormap(mcolors.TABLEAU_COLORS.keys())

    legend = [0] * embeddings.shape[0]
    for i in range(embeddings.shape[0] - 1, -1, -1):
        for j in range(embeddings.shape[1]):
            legend[i] = ax.bar(j, sort[i, j], width=0.8, color=cmap(asort[i, j]))

    plt.title(title)
    plt.ylabel('Mean Values')

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(legend, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)
    plt.show()


def plot_model_parameters_comparison(paths):
    import matplotlib.colors as mcolors
    representations = ["mean", "max", "min"]

    state_dicts = []
    labels = []
    for path in paths:
        model = MeshNetwork()
        model.load(path)
        state_dicts.append(model.state_dict())
        labels.append(path.split("/")[-3])
    SMALL_SIZE = 4
    MEDIUM_SIZE = 6
    BIGGER_SIZE = 12

    fig = plt.figure(figsize=(12.8, 7.2), dpi=300)
    axs = fig.subplots(len(state_dicts[0]), len(representations), sharex="none", sharey="none")
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    legend = [0] * len(state_dicts)
    for layer_id in range(len(list(state_dicts[0].keys()))):
        for repr_id in range(len(representations)):
            weights = []
            layer = list(state_dicts[0].keys())[layer_id]
            repr_method = getattr(np, representations[repr_id])
            for state_dict in state_dicts:
                weights.append(repr_method(state_dict[layer].numpy(), axis=0))
            weights = np.array(weights)

            asort = np.argsort(weights, axis=0)
            sort = np.sort(weights, axis=0)
            cmap = matplotlib.colors.ListedColormap(mcolors.TABLEAU_COLORS.keys())
            for i in range(weights.shape[0] - 1, -1, -1):
                for j in range(weights.shape[1]):
                    legend[i] = axs[layer_id, repr_id].bar(j, sort[i, j], width=0.8, color=cmap(asort[i, j]))
            axs[layer_id, repr_id].set_title(f"{representations[repr_id]}")

    layer_id = 0
    for ax in axs.flat:
        ax.tick_params(axis='both', which='major', labelsize=SMALL_SIZE)
        ax.tick_params(axis='both', which='minor', labelsize=SMALL_SIZE)
        if layer_id % (len(representations)) == 0:
            layer = list(state_dicts[0].keys())[int(layer_id / len(representations))]
            ax.set_ylabel(f"{layer.replace('.weight', '')}", rotation=45, labelpad=20)
        layer_id += 1
    fig.supxlabel('Weights Representation')
    fig.supylabel('Layers')

    # Put a legend below current axis

    fig.legend(legend, labels, loc="upper center", ncol=2)
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    plt.savefig("model_parameters_statistics.png")
    plt.close()


def plot_embeddings(model, dataset, device, filename=None):
    embeddings = np.empty((0, model.readout_dim))
    embeddings_labels = np.empty(0, dtype=np.int32)

    dataloader = GraphDataLoader(dataset, batch_size=1, drop_last=False)

    sampler = 0
    for graph, label in dataloader:
        tmp = model(graph, dataset.graphs[sampler].patches, device)[1]
        embeddings = torch.vstack((embeddings, tmp))
        embeddings_labels = np.hstack((embeddings_labels, np.tile(label.detach().cpu().numpy(), tmp.shape[0])))
        sampler += 1

    embeddings = embeddings.detach().cpu().numpy()
    embeddings = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(embeddings)

    palette = np.array(sns.color_palette("hls", len(np.unique(embeddings_labels))))
    f, ax = plt.subplots()
    for c in np.unique(embeddings_labels):
        ax.scatter(embeddings[embeddings_labels == c, 0], embeddings[embeddings_labels == c, 1], label=c, color=palette[c])
    ax.legend(title="Classes")
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename + "_EmbSpace.png")

    fig = plt.figure()
    axes = fig.subplots(5, 3, sharex=True, sharey=True)
    flat_axes = axes.flat
    for i, c in enumerate(np.unique(embeddings_labels)):
        flat_axes[i].scatter(embeddings[embeddings_labels == c, 0], embeddings[embeddings_labels == c, 1], label=c, color=palette[c])
    # ax.legend(title="Classes")
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename="_Embeddings.png")


def print_weights(model):
    named_parameters = model.named_parameters()
    for name, layer in named_parameters:
        print(f"Layer: {name}   {layer}")


def print_weights_difference(model, precedent_weights):
    named_parameters = model.named_parameters()
    names = []
    current_parameters = []
    old_parameters = []
    print(id(named_parameters))
    print(id(precedent_weights))
    for name, layer in named_parameters:
        names.append(name)
        current_parameters.append(layer)

    for name, layer in precedent_weights:
        old_parameters.append(layer)

    for i, name in enumerate(names):
        print(f"Layer: {name}   {old_parameters[i] - current_parameters[i]}")
