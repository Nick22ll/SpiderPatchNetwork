import dgl
import torch.nn as nn

from Networks.GATSpiderPatchModules import GATNodeWeigher
from Networks.SpiralReadout import SpiralReadout
from Networks.UniversalReadout import UniversalReadout


class SPReader(nn.Module):
    def __init__(self, feat_in_dim, readout_function="AR", weigher_mode=None, weights_in_dim=0, *args, **kwargs):
        super(SPReader, self).__init__()

        ####  Readouts Utils  ####
        self.weigher_mode = weigher_mode

        if self.weigher_mode is not None:
            if self.weigher_mode == "sp_weights":
                self.weigher = nn.Identity()
            elif self.weigher_mode == "attn_weights":
                self.weigher = GATNodeWeigher(feat_in_dim, weights_in_dim, False)
            elif self.weigher_mode == "attn_weights+feats":
                self.weigher = GATNodeWeigher(feat_in_dim, weights_in_dim, True)
            else:
                raise "Wrong Weigher Mode"

        #### Readout Layers  ####
        if readout_function == "UR":
            self.readout = UniversalReadout(feat_in_dim)
            self.embed_dim = self.readout.readout_dim
        elif readout_function == "AR":
            self.readout = dgl.mean_nodes
            self.embed_dim = feat_in_dim
        elif readout_function == "SR":
            self.readout = SpiralReadout(feat_in_dim * kwargs["sp_nodes"])
            self.embed_dim = self.readout.readout_dim
        else:
            raise "Unknown Readout Function"

    def forward(self, spider_patch, node_feats):

        spider_patch.ndata['updated_feats'] = node_feats

        if self.weigher_mode is not None:
            spider_patch.ndata['node_weights'] = self.weigher(spider_patch, "aggregated_weights", "updated_feats")
            weights_name = 'node_weights'
        else:
            weights_name = None

        return self.readout(spider_patch, 'updated_feats', weights_name)
