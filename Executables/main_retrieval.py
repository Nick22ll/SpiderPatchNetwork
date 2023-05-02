import os
import sys

import numpy as np
from dgl.dataloading import GraphDataLoader

from CSIRS.CSIRS import CSIRSv2Spiral
from SpiderDatasets.RetrievalDataset import generateMesh, generateLabels, RetrievalDataset
from SpiderPatch.SpiderPatch import SPMatrixDistanceV1


def main():
    # generateRetrievalMesh("face3")
    dataset = RetrievalDataset(dataset_name="face3")
    # dataset.generate("../Retrieval/Meshes/face3.pkl", "../Retrieval/Labels/face3.pkl", 5, 4, 6, True, CSIRS_type=CSIRSv2Spiral)
    dataset.load_from("../Retrieval/Datasets", "face3")

    features = np.array(["gauss_curvature", "mean_curvature", "curvedness", "K2", "local_depth"])
    features_to_keep = [0, 1, 2, 3, 4]
    radius_to_keep = [0, 1, 2, 3, 4]

    dataset.keepCurvaturesResolution(radius_to_keep)
    dataset.aggregateNodeFeatures(features[features_to_keep], "aggregated_feats")
    dataset.removeNonAggregatedFeatures()

    dataloader = GraphDataLoader(dataset.graphs[:1000], batch_size=1000, drop_last=False)  # patches[random_sequence]
    for batched_SP in dataloader:
        distance = SPMatrixDistanceV1(batched_SP, "aggregated_feats")
    print()


def generateRetrievalMesh(mesh_name):
    generateMesh(f"../../MeshDataset/SHREC18/shrec_retrieval_tortorici/{mesh_name}.off", f"{mesh_name}")
    generateLabels(f"{mesh_name}", f"../../MeshDataset/SHREC18/shrec_retrieval_tortorici/Labels/{mesh_name}.mat")


if __name__ == "__main__":
    # Change the scripts working directory to the script's own directory
    os.chdir(os.path.dirname(sys.argv[0]))
    main()
