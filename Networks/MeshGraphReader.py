import dgl
import torch
from dgl.dataloading import GraphDataLoader
from torch import nn


class MeshGraphReader(nn.Module):
    def __init__(self, spider_patch_embedder=None):
        super(MeshGraphReader, self).__init__()

        self.spider_patch_embedder = spider_patch_embedder
        self.BATCH_SIZE = 512

    def forward(self, mesh_graph, batched_patches, device):
        if self.spider_patch_embedder is not None:
            BATCH_SIZE = self.BATCH_SIZE if len(batched_patches) >= self.BATCH_SIZE else len(batched_patches)
            SP_embeddings = torch.empty((len(batched_patches), self.spider_patch_embedder.embed_dim), device=device)
            dataloader = GraphDataLoader(batched_patches, batch_size=BATCH_SIZE, drop_last=False)  # patches[random_sequence]
            for idx, SP_batch in enumerate(dataloader):
                SP_embeddings[(idx * BATCH_SIZE):((idx * BATCH_SIZE) + SP_batch.batch_size)] = self.spider_patch_embedder(SP_batch, SP_batch.ndata["aggregated_feats"])
            mesh_graph.ndata['updated_feats'] = SP_embeddings

        return dgl.mean_nodes(mesh_graph, 'updated_feats')
