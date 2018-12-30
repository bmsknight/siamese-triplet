
from mydataset import  SSDataset

import torch
from torch.optim import lr_scheduler
import torch.optim as optim

from trainer import fit
import numpy as np
cuda = torch.cuda.is_available()

import matplotlib.pyplot as plt

batch_size = 32

train_dataset = SSDataset(train=True)
test_dataset = SSDataset(train=False)
kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

n_classes = 10

mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']


def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    # plt.figure(figsize=(10,10))
    for i in range(10):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mnist_classes)


def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 32))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels

from datasets import BalancedBatchSampler

# We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
train_batch_sampler = BalancedBatchSampler(train_dataset, n_classes=10, n_samples=5)
test_batch_sampler = BalancedBatchSampler(test_dataset, n_classes=10, n_samples=5)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

# Set up the network and training parameters
from networks import EmbeddingNet
from losses import OnlineContrastiveLoss
from utils import AllPositivePairSelector, HardNegativePairSelector # Strategies for selecting pairs within a minibatch

margin = 1.
embedding_net = EmbeddingNet()
model = embedding_net
if cuda:
    model.cuda()
loss_fn = OnlineContrastiveLoss(margin, HardNegativePairSelector())
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 10, gamma=0.5, last_epoch=-1)
n_epochs = 30
log_interval = 100

fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)


train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, model)
val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, model)
train_length = train_embeddings_cl.shape[0]
dataset_complete = np.vstack((train_embeddings_cl, val_embeddings_cl))

from sklearn.manifold import TSNE

n_sne = 7000

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

tsne_dataset_complete = tsne.fit_transform(dataset_complete)
tsne_train_embeddings_cl = tsne_dataset_complete[:train_length, :]
tsne_val_embeddings_cl = tsne_dataset_complete[train_length:, :]

plt.subplot(1, 2, 1)
# plot_embeddings(train_embeddings_cl, train_labels_cl)
plot_embeddings(tsne_train_embeddings_cl, train_labels_cl)

plt.subplot(1, 2, 2)
# plot_embeddings(val_embeddings_cl, val_labels_cl)
plot_embeddings(tsne_val_embeddings_cl, val_labels_cl)
plt.show()

