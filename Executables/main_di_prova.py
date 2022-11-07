import os
import pickle

import multiprocessing
import os
import re
import pickle
import random
import warnings

import numpy as np
import pickle as pkl

from mayavi.tools.helper_functions import quiver3d
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from Executables.main_DatasetGenerator import generateMeshGraphDatasetFromPatches
from Mesh import *
from math import floor

from Mesh.Mesh import Mesh
from SHREC_Utils import subdivide_for_mesh
from CSIRS.CSIRS import CSIRSv2, CSIRSv2Spiral
from SpiderDatasets.PatchDatasetGenerator import generateSPDatasetFromConcRings
from SpiderPatch.SpiderPatch import SpiderPatchLRF
from SpiderPatch.SuperPatch import SuperPatch_preloaded


def main():
    # with open(f"U:\AssegnoDiRicerca\PythonProject\Datasets\SpiderPatches\SHREC17_R10_R4_P6_CSIRSv2Spiral\class_0\id_0/resolution_level_0\spiderPatches609.pkl", "rb") as mesh_file:
    #     concRingsA = pkl.load(mesh_file)

    with open("U:\AssegnoDiRicerca\PythonProject\Retrieval\Datasets\SHREC17_RR10R4P6_CSIRSv2Spiral_HOMOGENEOUS3.pkl", "rb") as file:
        dataset = pkl.load(file)
        for graph in dataset.graphs:
            with open("U:\AssegnoDiRicerca\PythonProject\Datasets\Meshes\SHREC17\class_3\id_36/resolution_level_3\mesh280.pkl", "rb") as file:
                mesh = pkl.load(file)
            mesh.drawWithSpiderPatches([graph])
    #
    # with open("U:\AssegnoDiRicerca\PythonProject\Datasets\Meshes\SHREC17\class_3\id_36/resolution_level_3\mesh280.pkl", "rb") as file:
    #     mesh = pkl.load(file)
    # with open("U:\AssegnoDiRicerca\PythonProject\Retrieval\Datasets\SHREC17_R10R4P6_CSIRSv2Spiral_HOMOGENEOUS3.pkl", "rb") as file:
    #     dataset = pkl.load(file)
    #     for graph in dataset.graphs:
    #         mesh.drawWithSpiderPatches([graph])

    with open(f"U:\AssegnoDiRicerca\PythonProject\Datasets\ConcentricRings\SHREC17_RR10_R4_P6_CSIRSv2Spiral\class_1\id_17/resolution_level_3\concRing519.pkl", "rb") as file:
        concRings = pickle.load(file)
    with open(f"U:\AssegnoDiRicerca\PythonProject\Datasets\Meshes\SHREC17\class_1\id_17/resolution_level_3\mesh519.pkl", "rb") as mesh_file:
        mesh = pickle.load(mesh_file)
    for concRing in concRings:
        mesh.drawWithConcRings(concRing)

    load_path = "U:\AssegnoDiRicerca\PythonProject\Datasets\ConcentricRings\SHREC17_R10_RI4_P6_CSIRSv2Spiral"
    for label in os.listdir(load_path):
        for id in os.listdir(f"{load_path}/{label}"):
            for resolution_level in os.listdir(f"{load_path}/{label}/{id}"):
                conc_filename = os.listdir(f"{load_path}/{label}/{id}/{resolution_level}")[0]
                with open(f"{load_path}/{label}/{id}/{resolution_level}/{conc_filename}", "rb") as file:
                    concRings = pickle.load(file)
                with open(f"U:\AssegnoDiRicerca\PythonProject\Datasets\Meshes\SHREC17/{label}/{id}/{resolution_level}/{conc_filename.replace('concRing', 'mesh')}", "rb") as mesh_file:
                    mesh = pickle.load(mesh_file)
                for concRing in concRings[0:10]:
                    mesh.drawWithConcRings(concRing)


if __name__ == "__main__":
    main()
