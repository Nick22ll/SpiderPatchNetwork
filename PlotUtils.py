
import itertools
import os
import pickle

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from dgl.dataloading import GraphDataLoader
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from tqdm import tqdm
import pickle as pkl
from natsort import os_sorted


def plot_confusion_matrix(cm, target_names=None, cmap_name=None):
    if cmap_name is None:
        cmap = plt.get_cmap('Blues')
    else:
        cmap = plt.get_cmap(cmap_name)

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


def save_confusion_matrix(cm, path, filename, target_names=None, cmap=None):
    os.makedirs(path, exist_ok=True)

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
    plt.savefig(path + "/" + filename)
    plt.close()
    plt.clf()


def plot_grad_flow(named_parameters, verbose=1, legend=False):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    # plt.rcParams['figure.dpi'] = 300
    # plt.rcParams["figure.figsize"] = (plt.rcParamsDefault["figure.figsize"][0] * 2,  plt.rcParamsDefault["figure.figsize"][1])

    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.grad is None:
            print(f"Parameter {n} is none!")
        else:
            if p.requires_grad and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu().detach().numpy())
                max_grads.append(p.grad.abs().max().cpu().detach().numpy())
                if ave_grads[-1] == 0 or max_grads[-1] == 0 and verbose > 0:
                    print(f"Paameter {n} is 0!")
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.4, lw=1, color="darkorange")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=1, color="lime")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    layers = [layers[i].replace(".weight", "") for i in range(len(layers))]
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation=90)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=np.max(ave_grads) / 2)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    if legend:
        plt.legend([Line2D([0], [0], color="deepskyblue", lw=4),
                    Line2D([0], [0], color="steelblue", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    plt.show()
    plt.close()

    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
    plt.rcParams['figure.dpi'] = plt.rcParamsDefault['figure.dpi']


def plot_training_statistics(path, filename, epochs, losses, val_accuracies, val_epochs, val_losses=None, title="Training Statistics"):
    os.makedirs(path, exist_ok=True)
    plt.plot(epochs, losses, 'sienna', label='Training Loss')
    plt.plot(val_epochs, val_accuracies, 'steelblue', label='Validation Accuracy')
    if val_losses is not None:
        plt.plot(val_epochs, val_losses, 'cyan', label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path + "/" + filename)
    plt.close()


def func_key(elem):
    return int(elem[0][0:elem[0].find("_")])


def plot_voting_confidences(dict_to_plot, path, filename):
    confidences = []
    classes = []

    for key, value in list(dict_to_plot["data"].items()):
        classes.append(str(key) + "_" + str(value[0]))
        confidences.append(value[1])

    classes, confidences = zip(*sorted(zip(classes, confidences), key=func_key, reverse=False))

    # creating the bar plot
    plt.figure(figsize=(20, 8))
    plt.bar(classes, confidences, color=dict_to_plot["colors"], width=0.8)
    plt.ylim(bottom=0, top=1)
    plt.grid(True)
    plt.xticks(rotation=60)
    plt.xlabel("Classes")
    plt.ylabel("Confidence")
    plt.title("Top 3 Confidence levels")
    plt.tight_layout()
    plt.savefig(path + "/" + filename)
    plt.close()


def save_embeddings_statistics(path, embeddings, id):
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/patch_embedding_Epoch{id}.pkl", "wb") as embeddings_file:
        pickle.dump(embeddings, embeddings_file, protocol=-1)


def load_embeddings_statistics(path, id):
    with open(f"{path}/patch_embedding_Epoch{id}.pkl", "rb") as embeddings_file:
        return pickle.load(embeddings_file)


def plot_embeddings_statistics(paths, id, title="", labels=None):


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
    cmap = mcolors.ListedColormap(mcolors.TABLEAU_COLORS.keys())

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


def plot_data_distributions(dataset, mask, feature_names=None):
    if feature_names is None:
        feature_names = ["aggregated_feats"]

    to_plot = {}
    for feature in feature_names:
        tmp = []
        for idx, mesh_graph in tqdm(enumerate(dataset.graphs[mask])):
            for spider_patch in mesh_graph.patches:
                list1 = spider_patch.ndata[feature].tolist()
                tmp.extend(list1)
        to_plot[feature] = np.array(tmp)

    for feature in feature_names:
        for col in range(to_plot[feature].shape[1]):
            print(f"feature {feature}[{col}] - > {min(to_plot[feature][:, col])}", max(to_plot[feature][:, col]))

    for feature in feature_names:
        fig, axes = plt.subplots(nrows=1, ncols=to_plot[feature].shape[1], sharex=False, sharey=False)
        for ax, col in zip(axes, range(to_plot[feature].shape[1])):
            sns.kdeplot(data=to_plot[feature][:, col], ax=ax)
            ax.set(title=f'Distribution of Feature: {feature}', xlabel=None)
        fig.tight_layout()
        plt.show()


def plotCVConfusionMatrix(CV_path, cmap_name=None):
    # open matrices
    matrices = []
    for experiment in os_sorted(os.listdir(CV_path)):
        with open(f"{CV_path}/{experiment}/MeshNetworkBestAcc/MeshGraphConfusionMatrices.pkl", "rb") as matrices_file:
            matrices.append(pkl.load(matrices_file))

    sum_matrix = np.zeros_like(matrices[0][0])
    voting_matrix = np.zeros_like(matrices[0][0])
    for matrix in matrices:
        sum_matrix += matrix[1]
        voting_matrix[matrix[0] > 0.50] += 1

    plot_confusion_matrix(sum_matrix.astype(int), cmap_name=cmap_name)

    sum_matrix /= sum_matrix.astype(np.float).sum(axis=1) * 0.01

    plot_confusion_matrix(sum_matrix, cmap_name=cmap_name)

    voting_matrix /= voting_matrix.astype(np.float).sum(axis=1) * 0.01

    plot_confusion_matrix(voting_matrix, cmap_name=cmap_name)

    print(f"Voting accuracy: {np.trace(voting_matrix) / voting_matrix.shape[0]}")
    print(f"NO Voting accuracy: {np.trace(sum_matrix) / sum_matrix.shape[0]}")
