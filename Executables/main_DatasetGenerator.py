import os
import sys

import numpy as np

from CSIRS.CSIRS import CSIRSv2Spiral
from SpiderDatasets.ConcRingsDatasetGenerator import parallelGenerateConcRingDataset
from SpiderDatasets.MeshDatasetGenerator import parallelGenerateMeshDataset, generateMeshDataset
from SpiderDatasets.MeshGraphDataset import MeshGraphDataset
from SpiderDatasets.PatchDatasetGenerator import generateSPDatasetFromConcRings
from SpiderDatasets.SpiralMehGraphDataset import SpiralMeshGraphDataset
from SpiderDatasets.SpiralMeshGraphDatasetForNNTraining import SpiralMeshGraphDatasetForNNTraining


def generateMeshGraphDatasetFromPatches(patch_path, graph_per_mesh, patch_per_graph, resolution_level, neighbours_number, features_to_keep, dataset_name):
    dataset = MeshGraphDataset(dataset_name=dataset_name)
    dataset.fromRawPatches(patch_path, graph_for_mesh=graph_per_mesh, patch_for_graph=patch_per_graph, connection_number=neighbours_number, feature_to_keep=features_to_keep, only_resolution_level=resolution_level)
    dataset.save_to(f"Datasets/MeshGraphs")


def generateGeoMeshGraphDatasetFromPatches(patch_path, graph_per_mesh, patch_per_graph, resolution_level, neighbours_number, features_to_keep, dataset_name):
    dataset = MeshGraphDataset(dataset_name=dataset_name)
    dataset.fromRawPatchesGeo(patch_path, graph_for_mesh=graph_per_mesh, patch_for_graph=patch_per_graph, connection_number=neighbours_number, feature_to_keep=features_to_keep, only_resolution_level=resolution_level)
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
    # to_exclude = []
    # load_path = f"U:\AssegnoDiRicerca\PythonProject\Datasets/NormalizedMeshes\SHREC17"
    # # load_path = f"U:\AssegnoDiRicerca\PythonProject\Datasets\ConcentricRings\SHREC17_R0.1_R4_P6_CSIRSv2Spiral"
    # for label in os.listdir(load_path):
    #     for sample_id in os.listdir(f"{load_path}/{label}"):
    #         to_exclude.append(int(re.sub(r"\D", "", sample_id)))
    # #
    # to_extract = []
    # load_path = f"U:\AssegnoDiRicerca\PythonProject\Datasets\Meshes\SHREC17"
    # for label in os.listdir(load_path):
    #     for sample_id in os.listdir(f"{load_path}/{label}"):
    #         to_extract.append(int(re.sub(r"\D", "", sample_id)))
    #
    # to_extract = [el for el in to_extract if el not in to_exclude]
    # parallelGenerateMeshDataset(from_mesh_dataset="SHREC20", to_extract="all", normalize=False, resolution_level=None, num_threads=8)

    to_extract = np.array(np.load("Executables/samples2.npy"), dtype=int)
    generateMeshDataset(dataset_name="SHREC17_prova", from_mesh_dataset="SHREC17", samples_to_extract=to_extract[14:], normalize=False, perturbe=(1, 0.2), resolution_level="level_3")
    # parallelGenerateMeshDataset(dataset_name="SHREC17PERTURBED", from_mesh_dataset="SHREC17", samples_to_extract=to_extract, normalize=False, perturbe=(1, 0.2), resolution_level="level_3")

    # generateSPDatasetFromConcRings("Datasets/ConcentricRings/SHREC17_R0.05_R6_P8_CSIRSv2Spiral", valid_rings=2)
    # generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R15_R4_P6_CSIRSv2Spiral", 50, 25, "level_3", 2, None, "SHREC17_R15_R4_P68_CSIRSv2Spiral_MGRAPHS50_SPIDER25_CONN2_RES3")
    # generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R0.05_R6_P8_CSIRSv2Spiral", 25, 75, "level_3", 5, None, "SHREC17_R0.05_R6_P8_CSIRSv2Spiral_MGRAPHS25_SPIDER75_CONN5_RES3")
    # generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R10_R4_P6_CSIRSv2Spiral", 20, 100, "level_3", 0, None, "SHREC17_R10_R4_P6_CSIRSv2Spiral_MGRAPHS20_SPIDER100_CONNFC_RES3")
    # generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R10_R4_P6_CSIRSv2Spiral", 40, 25, "level_3", 5, None, "SHREC17_R10_R4_P6_CSIRSv2Spiral_MGRAPHS40_SPIDER25_CONN5_RES3")
    # generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R10_R4_P6_CSIRSv2Spiral", 40, 25, "level_3", 0, None, "SHREC17_R10_R4_P6_CSIRSv2Spiral_MGRAPHS40_SPIDER25_FC_RES3")

    # to_exclude = []
    # load_path = f"U:\AssegnoDiRicerca\PythonProject\Datasets\ConcentricRings\SHREC17_R0.1_R6_P8_CSIRSv2Spiral"
    # for label in os.listdir(load_path):
    #     for sample_id in os.listdir(f"{load_path}/{label}"):
    #         for resolution_level in os.listdir(f"{load_path}/{label}/{sample_id}"):
    #             mesh_id = os.listdir(f"{load_path}/{label}/{sample_id}/{resolution_level}")[0]
    #             # file_bytes = os.path.getsize(f"{load_path}/{label}/{sample_id}/{resolution_level}/{mesh_id}")
    #             if resolution_level == "resolution_level_3":
    #                 # with open(f"{load_path}/{label}/{sample_id}/{resolution_level}/{mesh_id}", "rb") as mesh_file:
    #                 #     concRings = pickle.load(mesh_file)
    #                 #     if len(concRings) > 400:
    #                 to_exclude.append(int(re.sub(r"\D", "", mesh_id)))
    #
    # to_extract = []
    # load_path = f"U:\AssegnoDiRicerca\PythonProject\Datasets\Meshes\SHREC17"
    # for label in os.listdir(load_path):
    #     for sample_id in os.listdir(f"{load_path}/{label}"):
    #         for resolution_level in os.listdir(f"{load_path}/{label}/{sample_id}"):
    #             mesh_id = os.listdir(f"{load_path}/{label}/{sample_id}/{resolution_level}")[0]
    #             # if resolution_level == "resolution_level_0" or resolution_level == "resolution_level_1" or resolution_level == "resolution_level_2" or resolution_level == "resolution_level_3" and int(re.sub(r"\D", "", mesh_id)):  # not in to_exclude
    #             if resolution_level == "resolution_level_3":
    #                 to_extract.append(int(re.sub(r"\D", "", mesh_id)))
    #                 # with open(f"{load_path}/{label}/{sample_id}/{resolution_level}/{mesh_id}", "rb") as mesh_file:
    #                 #     mesh = pickle.load(mesh_file)
    #                 #     print(f"Class: {label} ID: {mesh_id}, BB: {mesh.oriented_bounding_box['extent']}")
    #
    # to_extract = [el for el in to_extract if el not in to_exclude]

    # parallelGenerateMeshDataset("all", True, "level_0")
    # parallelGenerateMeshDataset("all", True, "level_1")
    # parallelGenerateMeshDataset("all", True, "level_2")

    # configurations = [{"radius": 10, "rings": 6, "points": 8, "CSIRS_type": CSIRSv2Spiral, "relative_radius": False}]
    # parallelGenerateConcRingDataset(dataset_name="SHREC17PERTURBED", to_extract=None, configurations=configurations, use_normalized_meshs=False, num_thread=12)
    #
    # configurations = [{"radius": 0.15, "rings": 6, "points": 8, "CSIRS_type": CSIRSv2Spiral, "relative_radius": False}, {"radius": 0.1, "rings": 6, "points": 8, "CSIRS_type": CSIRSv2Spiral, "relative_radius": False}]
    # parallelGenerateConcRingDataset(dataset_name="SHREC20", to_extract=None, configurations=configurations, use_normalized_meshs=True, num_thread=12)
    #
    # parallelGenerateConcRingDataset(to_extract, configurations=numpy.array([[10, 6, 8], [10,4,6]]), relative_radius=False, normalized=False)
    # parallelGenerateConcRingDataset(to_extract, configurations=numpy.array([[6, 4, 6], [15,4,6]]), relative_radius=False, normalized=False)
    # parallelGenerateConcRingDataset(to_extract, configurations=numpy.array([[0.1, 4, 6]]), relative_radius=False, normalized=True)
    #

    # generateSPDatasetFromConcRings("Datasets/ConcentricRings/SHREC17PERTURBED_R10_R6_P8_CSIRSv2Spiral", valid_rings=2)
    generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17PERTURBED_R10_R6_P8_CSIRSv2Spiral", 25, 50, "level_3", 10, None, "SHREC17PERTURBED_R10_R6_P8_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN10")
    # generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC20_R0.1_R6_P8_CSIRSv2Spiral", 25, 50, "all", 10, None, "SHREC20_R0.1_R6_P8_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN10")
    # generateGeoMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC20_R0.05_R6_P8_CSIRSv2Spiral", 5, 20, "all", 5, None, "SHREC20_R0.05_R6_P8_CSIRSv2Spiral_MGRAPHS5_SPIDER20_CONN5_GEO")
    # generateGeoMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC20_R0.15_R6_P8_CSIRSv2Spiral", 25, 25, "all", 5, None, "SHREC20_R0.15_R6_P8_CSIRSv2Spiral_MGRAPHS25_SPIDER25_CONN5_GEO")
    # generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R0.1_R6_P8_CSIRSv2Spiral", 25, 25, "level_3", 5, None, "SHREC17_R0.1_R6_P8_CSIRSv2Spiral_MGRAPHS25_SPIDER25_CONN5_RES3")


#
# generateSPDatasetFromConcRings("Datasets/ConcentricRings/SHREC17_R6_R4_P6_CSIRSv2Spiral", valid_rings=2, normalized=False)
# generateSPDatasetFromConcRings("Datasets/ConcentricRings/SHREC17_R15_R4_P6_CSIRSv2Spiral", valid_rings=2, normalized=False)
# generateSPDatasetFromConcRings("Datasets/ConcentricRings/SHREC17_R0.1_R4_P6_CSIRSv2Spiral", valid_rings=2, normalized=True)
# generateSPDatasetFromConcRings("Datasets/ConcentricRings/SHREC17_R10_R6_P8_CSIRSv2Spiral", valid_rings=2, normalized=False)
# generateSPDatasetFromConcRings("Datasets/ConcentricRings/SHREC17_R10_R4_P6_CSIRSv2Spiral", valid_rings=2, normalized=False)

# generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R6_R4_P6_CSIRSv2Spiral", 6, 35, "all", 5, None, "SHREC17_R6_R4_P6_CSIRSv2Spiral_MGRAPHS6_SPIDER35_CONN5_RESALL")
# generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R10_R4_P6_CSIRSv2Spiral", 10, 20, "all", 3, None, "SHREC17_R10_R4_P6_CSIRSv2Spiral_MGRAPHS10_SPIDER20_CONN3_RESALL")
# generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R15_R4_P6_CSIRSv2Spiral", 25, 50, "level_3", 5, None, "SHREC17_R15_R4_P6_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN5_RES3")
# generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R10_R6_P8_CSIRSv2Spiral", 1, 25, "level_3", 10, None, "SHREC17_R10_R6_P8_CSIRSv2Spiral_MGRAPHS1_SPIDER25_CONN10_RES3")
# generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R10_R6_P8_CSIRSv2Spiral", 25, 50, "level_3", 10, None, "SHREC17_R10_R6_P8_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN10_RES3")
# generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R0.1_R6_P8_CSIRSv2Spiral", 25, 50, "level_3", 5, None, "SHREC17_R0.1_R6_P8_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN5_RES3")
# generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R0.1_R6_P8_CSIRSv2Spiral", 25, 50, "level_3", 10, None, "SHREC17_R0.1_R6_P8_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN10_RES3")

# generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R10_R6_P8_CSIRSv2Spiral", 25, 50, "level_0", 10, None, "SHREC17_R10_R6_P8_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN10_RES0")
# generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R10_R6_P8_CSIRSv2Spiral", 25, 50, "level_1", 10, None, "SHREC17_R10_R6_P8_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN10_RES1")
# generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R10_R6_P8_CSIRSv2Spiral", 25, 50, "level_2", 10, None, "SHREC17_R10_R6_P8_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN10_RES2")
# generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R10_R6_P8_CSIRSv2Spiral", 25, 50, "level_3", 10, None, "SHREC17_R10_R6_P8_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN10_RES3")

# generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R10_R6_P8_CSIRSv2Spiral", 25, 50, "all", 10, None, "SHREC17_R10_R6_P8_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN10_ALLRES")

# generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R10_R4_P6_CSIRSv2Spiral", 25, 50, "level_0", 5, None, "SHREC17_R10_R4_P6_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN5_RES0")
# generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R10_R4_P6_CSIRSv2Spiral", 25, 50, "level_1", 5, None, "SHREC17_R10_R4_P6_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN5_RES1")
# generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R10_R4_P6_CSIRSv2Spiral", 25, 50, "level_2", 5, None, "SHREC17_R10_R4_P6_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN5_RES2")
# generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R10_R4_P6_CSIRSv2Spiral", 25, 50, "level_3", 5, None, "SHREC17_R10_R4_P6_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN5_RES3")
# generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R10_R4_P6_CSIRSv2Spiral", 25, 50, "all", 5, None, "SHREC17_R10_R4_P6_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN5_ALLRES")
#
# generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R0.1_R6_P8_CSIRSv2Spiral", 25, 50, "level_0", 5, None, "SHREC17_R0.1_R6_P8_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN5_RES0")
# generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R0.1_R6_P8_CSIRSv2Spiral", 25, 50, "level_1", 5, None, "SHREC17_R0.1_R6_P8_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN5_RES1")
# generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R0.1_R6_P8_CSIRSv2Spiral", 25, 50, "level_2", 5, None, "SHREC17_R0.1_R6_P8_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN5_RES2")
# generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R0.1_R6_P8_CSIRSv2Spiral", 25, 50, "level_3", 5, None, "SHREC17_R0.1_R6_P8_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN5_RES3")
# generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R0.1_R6_P8_CSIRSv2Spiral", 25, 50, "all", 5, None, "SHREC17_R0.1_R6_P8_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN5_ALLRES")
#
# generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R0.1_R4_P6_CSIRSv2Spiral", 25, 50, "level_0", 5, None, "SHREC17_R0.1_R4_P6_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN5_RES0")
# generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R0.1_R4_P6_CSIRSv2Spiral", 25, 50, "level_1", 5, None, "SHREC17_R0.1_R4_P6_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN5_RES1")
# generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R0.1_R4_P6_CSIRSv2Spiral", 25, 50, "level_2", 5, None, "SHREC17_R0.1_R4_P6_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN5_RES2")
# generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R0.1_R4_P6_CSIRSv2Spiral", 25, 50, "level_3", 5, None, "SHREC17_R0.1_R4_P6_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN5_RES3")
# generateMeshGraphDatasetFromPatches("Datasets/SpiderPatches/SHREC17_R0.1_R4_P6_CSIRSv2Spiral", 25, 50, "all", 5, None, "SHREC17_R0.1_R4_P6_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN5_ALLRES")


if __name__ == "__main__":
    main()
