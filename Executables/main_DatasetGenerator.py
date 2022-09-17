import os
import re
from SpiderDatasets.ConcRingsDatasetGenerator import parallelGenerateConcRingDataset, generateConcRingDataset
from SpiderDatasets.MeshGraphDataset import MeshGraphDataset
from SpiderDatasets.MeshGraphForTrainingDataset import MeshGraphDatasetForNNTraining
from SpiderDatasets.SpiralMehGraphDataset import SpiralMeshGraphDataset
from SpiderDatasets.SpiralMeshGraphDatasetForNNTraining import SpiralMeshGraphDatasetForNNTraining


def generateMeshGraphDatasetFromPatches(patch_path, graph_per_mesh, patch_per_graph, resolution_level, neighbours_number, features_to_keep, dataset_name):
    dataset = MeshGraphDataset(dataset_name=dataset_name)
    dataset.fromRawPatches(patch_path, resolution_level=resolution_level, graph_for_mesh=graph_per_mesh, patch_for_graph=patch_per_graph, connection_number=neighbours_number, feature_to_keep=features_to_keep)
    dataset.save_to(f"Datasets/MeshGraphs/{dataset_name}")


def generateMeshGraphDatasetFromSuperPatches(patch_path, graph_per_mesh, patch_per_graph, connection_number, features_to_keep, dataset_name):
    dataset = MeshGraphDataset(dataset_name=dataset_name)
    dataset.fromRawSuperPatches(patch_path, graph_for_mesh=graph_per_mesh, patch_for_graph=patch_per_graph, connection_number=connection_number, feature_to_keep=features_to_keep)
    dataset.save_to(f"Datasets/MeshGraphs/{dataset_name}")


def generateSpiralMeshGraphDatasetFromConcRings(conc_config, graph_per_mesh, conc_per_graph, connection_number, features_to_keep, dataset_name):
    dataset = SpiralMeshGraphDataset(dataset_name=dataset_name)
    dataset.fromRawConcRings(conc_config, graph_for_mesh=graph_per_mesh, conc_for_graph=conc_per_graph, connection_number=connection_number, feature_to_keep=features_to_keep)
    dataset.save_to(f"Datasets/SpiralMeshGraphs/{dataset_name}")


def generateMeshGraphDatasetForNNTraining(mesh_dataset_path):
    dataset = MeshGraphDatasetForNNTraining()
    name = mesh_dataset_path.split("/")[-1]
    dataset.loadFromMeshGraphDataset(mesh_dataset_path, name)
    new_name = name + "_Normalized"
    dataset.normalize()
    dataset.normalize_validation_dataset()
    save_path = f"Datasets/MeshGraphsForTraining/{name[:name.rfind('_')].replace('_level', '')}/{new_name}"
    os.makedirs(save_path, exist_ok=True)
    dataset.save(save_path)


def generateSpiralMeshGraphDatasetForNNTraining(mesh_dataset_path):
    dataset = SpiralMeshGraphDatasetForNNTraining()
    name = mesh_dataset_path.split("/")[-1]
    dataset.loadFromSpiralMeshGraphDataset(mesh_dataset_path, name)
    new_name = name + "_Normalized"
    dataset.normalize()
    dataset.normalize_validation_dataset()
    save_path = f"Datasets/SpiralMeshGraphsForTraining/{name[:name.rfind('_')].replace('_level', '')}/{new_name}"
    os.makedirs(save_path, exist_ok=True)
    dataset.save(save_path)


def main():
    # to_extract = list(range(720))
    # load_path = f"U:\AssegnoDiRicerca\PythonProject\Datasets\ConcentricRings\SHREC17_R10_RI7_P12"
    # for label in os.listdir(load_path):
    #     for id in os.listdir(f"{load_path}/{label}"):
    #         for resolution_level in os.listdir(f"{load_path}/{label}/{id}"):
    #             mesh_id = os.listdir(f"{load_path}/{label}/{id}/{resolution_level}")[0]
    #             if int(re.sub(r"\D", "", mesh_id)) in to_extract:
    #                 to_extract.remove(int(re.sub(r"\D", "", mesh_id)))
    #
    # parallelGenerateConcRingDataset(to_extract)

    # generateSpiralMeshGraphDatasetFromConcRings("R7_RI4_P6", graph_per_mesh=20, conc_per_graph=20, connection_number=5, features_to_keep=None, dataset_name=f"SHREC17_R7_RI4_P6_SPIRAL20_SAMPLE20_CONN5")
    generateSpiralMeshGraphDatasetForNNTraining(f"Datasets/SpiralMeshGraphs/SHREC17_R7_RI4_P6_SPIRAL20_SAMPLE20_CONN5")

    generateSpiralMeshGraphDatasetFromConcRings("R5_RI4_P6", graph_per_mesh=20, conc_per_graph=20, connection_number=5, features_to_keep=None, dataset_name=f"SHREC17_R5_RI4_P6_SPIRAL20_SAMPLE20_CONN5")
    generateSpiralMeshGraphDatasetFromConcRings("R5_RI4_P12", graph_per_mesh=20, conc_per_graph=20, connection_number=5, features_to_keep=None, dataset_name=f"SHREC17_R5_RI4_P12_SPIRAL20_SAMPLE20_CONN5")
    generateSpiralMeshGraphDatasetFromConcRings("R10_RI4_P6", graph_per_mesh=20, conc_per_graph=20, connection_number=5, features_to_keep=None, dataset_name=f"SHREC17_R10_RI4_P6_SPIRAL20_SAMPLE20_CONN5")
    generateSpiralMeshGraphDatasetFromConcRings("R10_RI7_P12", graph_per_mesh=20, conc_per_graph=20, connection_number=5, features_to_keep=None, dataset_name=f"SHREC17_R10_RI7_P12_SPIRAL20_SAMPLE20_CONN5")

    generateSpiralMeshGraphDatasetForNNTraining(f"Datasets/SpiralMeshGraphs/SHREC17_R5_RI4_P6_SPIRAL20_SAMPLE20_CONN5")
    generateSpiralMeshGraphDatasetForNNTraining(f"Datasets/SpiralMeshGraphs/SHREC17_R5_RI4_P12_SPIRAL20_SAMPLE20_CONN5")
    generateSpiralMeshGraphDatasetForNNTraining(f"Datasets/SpiralMeshGraphs/SHREC17_R10_RI4_P6_SPIRAL20_SAMPLE20_CONN5")
    generateSpiralMeshGraphDatasetForNNTraining(f"Datasets/SpiralMeshGraphs/SHREC17_R10_RI7_P12_SPIRAL20_SAMPLE20_CONN5")


if __name__ == "__main__":
    main()
