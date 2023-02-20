import torch
import torch.nn as nn
from online_triplet_loss.losses import *


class TripletMGSP(nn.Module):
    def __init__(self, alpha=1, beta=1, gamma=1):
        super(TripletMGSP, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.SP_triplet = nn.TripletMarginLoss()

    def forward(self, pred_class, labels, MG_embedding, SP_embeddings, positive_SP_emb, negative_SP_emb):
        # normalize input embeddings
        MG_embedding = torch.nn.functional.normalize(MG_embedding)

        # normalize input embeddings
        SP_embeddings = torch.nn.functional.normalize(SP_embeddings)
        positive_SP_emb = torch.nn.functional.normalize(positive_SP_emb)
        negative_SP_emb = torch.nn.functional.normalize(negative_SP_emb)
        SP_loss = self.SP_triplet(SP_embeddings, positive_SP_emb, negative_SP_emb)
        return self.cross_entropy(pred_class, labels) + batch_hard_triplet_loss(labels, MG_embedding, margin=self.MG_margin) + SP_loss


class CETripletMG(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(CETripletMG, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

        self.alpha = alpha
        self.beta = beta

    def forward(self, pred_class, labels, MG_embedding):
        # normalize input embeddings
        MG_embedding = torch.nn.functional.normalize(MG_embedding)
        CE_loss = (self.alpha * self.cross_entropy(pred_class, labels))
        TRI_loss = (self.beta * batch_hard_triplet_loss(labels, MG_embedding, margin=1))
        return CE_loss + TRI_loss, CE_loss, TRI_loss


class TripletMG(nn.Module):
    def __init__(self):
        super(TripletMG, self).__init__()

    def forward(self, labels, MG_embedding):
        # normalize input embeddings
        MG_embedding = torch.nn.functional.normalize(MG_embedding)
        TRI_loss = batch_hard_triplet_loss(labels, MG_embedding, margin=0.5)
        return TRI_loss
