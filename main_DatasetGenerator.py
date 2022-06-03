from SpiderDatasets.MeshGraphDataset import MeshGraphDataset
from SpiderDatasets.MeshGraphForTrainingDataset import MeshGraphDatasetForNNTraining
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
    # generatePatchDataset(start_idx=144)
    for level in ["all"]:  # "level_0", "level_1", "level_2", "level_3"
        # generateMeshGraphDatasetFromPatches("Datasets/Patches/SHREC17_R5_RI4_P6", 50, 50, level, f"SHREC17_R5_RI4_P6_PATCH50_SAMPLE50_{level}")
        # generateMeshGraphDatasetFromPatches("Datasets/Patches/SHREC17_R7_RI4_P8", 50, 40, level, f"SHREC17_R7_RI4_P8_PATCH40_SAMPLE50_{level}")
        generateMeshGraphDatasetFromPatches("Datasets/Patches/SHREC17_R10_RI6_P6", 50, 30, level, f"SHREC17_R10_RI6_P6_PATCH30_SAMPLE50_{level}")
        # generateMeshGraphDatasetFromPatches("Datasets/Patches/SHREC17_R10_RI6_P6", 20, 5, level, f"SHREC17_R10_RI6_P6_PATCH5_SAMPLE20_{level}")

    for level in ["all"]:  # "level_0", , "level_1", "level_2", "level_3"
        # generateMeshGraphDatasetForNNTraining(f"Datasets/MeshGraphs/SHREC17_R5_RI4_P6_PATCH50_SAMPLE50_{level}")
        # generateMeshGraphDatasetForNNTraining(f"Datasets/MeshGraphs/SHREC17_R7_RI4_P8_PATCH40_SAMPLE50_{level}")
        generateMeshGraphDatasetForNNTraining(f"Datasets/MeshGraphs/SHREC17_R10_RI6_P6_PATCH30_SAMPLE50_{level}")
        # generateMeshGraphDatasetForNNTraining(f"Datasets/MeshGraphs/SHREC17_R10_RI6_P6_PATCH5_SAMPLE20_{level}")


if __name__ == "__main__":
    main()
