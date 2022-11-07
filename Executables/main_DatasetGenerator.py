import os
import pickle
import re

import numpy

from SpiderDatasets.ConcRingsDatasetGenerator import parallelGenerateConcRingDataset, generateConcRingDataset
from SpiderDatasets.MeshGraphDataset import MeshGraphDataset
from SpiderDatasets.PatchDatasetGenerator import generateSPDatasetFromConcRings
from SpiderDatasets.SpiralMehGraphDataset import SpiralMeshGraphDataset
from SpiderDatasets.SpiralMeshGraphDatasetForNNTraining import SpiralMeshGraphDatasetForNNTraining


def generateMeshGraphDatasetFromPatches(patch_path, graph_per_mesh, patch_per_graph, resolution_level, neighbours_number, features_to_keep, dataset_name):
    dataset = MeshGraphDataset(dataset_name=dataset_name)
    dataset.fromRawPatches(patch_path, resolution_level=resolution_level, graph_for_mesh=graph_per_mesh, patch_for_graph=patch_per_graph, connection_number=neighbours_number, feature_to_keep=features_to_keep)
    dataset.save_to(f"Datasets/MeshGraphs")


def generateMeshGraphDatasetFromSuperPatches(patch_path, graph_per_mesh, patch_per_graph, connection_number, features_to_keep, dataset_name):
    dataset = MeshGraphDataset(dataset_name=dataset_name)
    dataset.fromRawSuperPatches(patch_path, graph_for_mesh=graph_per_mesh, patch_for_graph=patch_per_graph, connection_number=connection_number, feature_to_keep=features_to_keep)
    dataset.save_to(f"Datasets/MeshGraphs/{dataset_name}")


def generateSpiralMeshGraphDatasetFromConcRings(conc_config, graph_per_mesh, conc_per_graph, connection_number, features_to_keep, dataset_name, resolution_level="all"):
    dataset = SpiralMeshGraphDataset(dataset_name=dataset_name)
    dataset.fromRawConcRings(conc_config, graph_for_mesh=graph_per_mesh, conc_for_graph=conc_per_graph, connection_number=connection_number, feature_to_keep=features_to_keep, resolution_level=resolution_level)
    dataset.save_to(f"Datasets/SpiralMeshGraphs/{dataset_name}")


def generateSpiralMeshGraphDatasetForNNTraining(mesh_dataset_path):
    dataset = SpiralMeshGraphDatasetForNNTraining()
    name = mesh_dataset_path.split("/")[-1]
    dataset.loadFromSpiralMeshGraphDataset(mesh_dataset_path, name)
    new_name = name + "_NONORM"
    # new_name = name + "_Normalized"
    # dataset.normalize()
    # dataset.normalize_validation_dataset()
    save_path = f"Datasets/SpiralMeshGraphsForTraining/{name[:name.rfind('_')].replace('_level', '')}/{new_name}"
    os.makedirs(save_path, exist_ok=True)
    dataset.save(save_path)


def main():
    # generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R15_R4_P6_CSIRSv2Spiral", 50, 25, "level_3", 2, None, "SHREC17_R15_R4_P68_CSIRSv2Spiral_MGRAPHS50_SPIDER25_CONN2_RES3")
    # generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R10_R4_P6_CSIRSv2Spiral", 5, 100, "level_3", 0 , None, "SHREC17_R10_R4_P6_CSIRSv2Spiral_MGRAPHS5_SPIDER100_FC_RES3")
    # generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R10_R4_P6_CSIRSv2Spiral", 20, 100, "level_3", 0, None, "SHREC17_R10_R4_P6_CSIRSv2Spiral_MGRAPHS20_SPIDER100_CONNFC_RES3")
    # generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R10_R4_P6_CSIRSv2Spiral", 40, 25, "level_3", 5, None, "SHREC17_R10_R4_P6_CSIRSv2Spiral_MGRAPHS40_SPIDER25_CONN5_RES3")
    # generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R10_R4_P6_CSIRSv2Spiral", 40, 25, "level_3", 0, None, "SHREC17_R10_R4_P6_CSIRSv2Spiral_MGRAPHS40_SPIDER25_FC_RES3")

    to_exclude = []
    load_path = f"U:\AssegnoDiRicerca\PythonProject\Datasets\ConcentricRings\SHREC17_R10_R8_P10_CSIRSv2Spiral"
    for label in os.listdir(load_path):
        for id in os.listdir(f"{load_path}/{label}"):
            for resolution_level in os.listdir(f"{load_path}/{label}/{id}"):
                mesh_id = os.listdir(f"{load_path}/{label}/{id}/{resolution_level}")[0]
                file_bytes = os.path.getsize(f"{load_path}/{label}/{id}/{resolution_level}/{mesh_id}")
                if resolution_level == "resolution_level_3":
                    with open(f"{load_path}/{label}/{id}/{resolution_level}/{mesh_id}", "rb") as mesh_file:
                        concRings = pickle.load(mesh_file)
                        if len(concRings) > 400:
                            to_exclude.append(int(re.sub(r"\D", "", mesh_id)))
    # #
    to_extract = []
    load_path = f"U:\AssegnoDiRicerca\PythonProject\Datasets\Meshes\SHREC17"
    for label in os.listdir(load_path):
        for id in os.listdir(f"{load_path}/{label}"):
            for resolution_level in os.listdir(f"{load_path}/{label}/{id}"):
                mesh_id = os.listdir(f"{load_path}/{label}/{id}/{resolution_level}")[0]
                if resolution_level == "resolution_level_3" and int(re.sub(r"\D", "", mesh_id)) not in to_exclude:
                    to_extract.append(int(re.sub(r"\D", "", mesh_id)))
                    # with open(f"{load_path}/{label}/{id}/{resolution_level}/{mesh_id}", "rb") as mesh_file:
                    #     mesh = pickle.load(mesh_file)
                    #     print(f"Class: {label} ID: {mesh_id}, BB: {mesh.oriented_bounding_box['extent']}")

    generateConcRingDataset(to_extract, configurations=numpy.array([[10, 8, 10]]), relative_radius=False)
    generateSPDatasetFromConcRings("Datasets/ConcentricRings/SHREC17_R10_R8_P10_CSIRSv2Spiral", )
    generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R10_R8_P10_CSIRSv2Spiral", 50, 25, "level_3", 2, None, "SHREC17_R10_R8_P10_CSIRSv2Spiral_MGRAPHS50_SPIDER25_CONN2_RES3")

    # parallelGenerateConcRingDataset(to_extract, configurations=numpy.array([[15, 4, 6]]), relative_radius=False)
    # parallelGenerateConcRingDataset(None, configurations=numpy.array([[10, 8, 10]]), relative_radius=False)
    # # #
    # # #


if __name__ == "__main__":
    main()
