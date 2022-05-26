from SpiderDatasets.MeshGraphDataset import MeshGraphDataset, MeshGraphDatasetForNNTraining
from SpiderPatch.PatchDatasetGenerator import *


def generateMeshGraphDatasetFromPatches(patch_path, graph_per_mesh, patch_per_graph, resolution_level, dataset_name):
    dataset = MeshGraphDataset(dataset_name=dataset_name)
    dataset.fromRawPatches(patch_path, resolution_level=resolution_level, graph_for_mesh=graph_per_mesh, patch_for_graph=patch_per_graph)
    dataset.save_to(f"Datasets/MeshGraphs/{dataset_name}")


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


def main():
    # generatePatchDataset(start_idx=121)
    for level in ["level_0", "all", "level_1", "level_2", "level_3"]:  #
        generateMeshGraphDatasetFromPatches("Datasets/Patches/SHREC17_R5_RI4_P6", 5, 15, level, f"SHREC17_R5_RI4_P6_PATCH15_SAMPLE5_{level}")
        generateMeshGraphDatasetFromPatches("Datasets/Patches/SHREC17_R7_RI4_P8", 5, 10, level, f"SHREC17_R7_RI4_P8_PATCH10_SAMPLE5_{level}")
        generateMeshGraphDatasetFromPatches("Datasets/Patches/SHREC17_R10_RI6_P6", 5, 5, level, f"SHREC17_R10_RI6_P6_PATCH5_SAMPLE5_{level}")

    for level in ["level_0"]:  # , "all", "level_1", "level_2", "level_3"
        generateMeshGraphDatasetForNNTraining(f"Datasets/MeshGraphs/SHREC17_R5_RI4_P6_PATCH15_SAMPLE5_{level}")
        generateMeshGraphDatasetForNNTraining(f"Datasets/MeshGraphs/SHREC17_R7_RI4_P8_PATCH10_SAMPLE5_{level}")
        generateMeshGraphDatasetForNNTraining(f"Datasets/MeshGraphs/SHREC17_R10_RI6_P6_PATCH5_SAMPLE5_{level}")


if __name__ == "__main__":
    main()
