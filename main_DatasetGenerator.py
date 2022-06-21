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
    # dataset = MeshGraphDatasetForNNTraining()
    # dataset.load(f"Datasets/MeshGraphsForTraining/SHREC17_R10_RI6_P6_PATCH50_SAMPLE2_FC/SHREC17_R10_RI6_P6_PATCH50_SAMPLE2_FC_all_Normalized", f"SHREC17_R10_RI6_P6_PATCH50_SAMPLE2_FC_all")
    # graph = dataset.train_dataset.graphs[22]
    # mesh = Mesh()
    # mesh.load(f"U:\AssegnoDiRicerca\MeshDataset\SHREC17\PatternDB/{graph.mesh_id}.off")
    # mesh.draw_with_MeshGraph(graph)

    # generatePatchDataset()
    for level in ["all"]:  # \, "level_1", "level_2", "level_3","level_0",
        # generateMeshGraphDatasetFromPatches("Datasets/Patches/SHREC17_R5_RI4_P6", 5, 25, level, f"SHREC17_R5_RI4_P6_PATCH25_SAMPLE5_FC_{level}")
        # generateMeshGraphDatasetForNNTraining(f"Datasets/MeshGraphs/SHREC17_R5_RI4_P6_PATCH25_SAMPLE5_FC_{level}")
        # generateMeshGraphDatasetFromPatches("Datasets/Patches/SHREC17_R7_RI4_P8", 10, 15, level, f"SHREC17_R7_RI4_P8_PATCH15_SAMPLE10_FC_{level}")
        # generateMeshGraphDatasetForNNTraining(f"Datasets/MeshGraphs/SHREC17_R7_RI4_P8_PATCH15_SAMPLE10_FC_{level}")
        # generateMeshGraphDatasetFromPatches("Datasets/Patches/SHREC17_R10_RI6_P6", 5, 10, level, f"SHREC17_R10_RI6_P6_PATCH10_SAMPLE5_FC_{level}")
        # generateMeshGraphDatasetForNNTraining(f"Datasets/MeshGraphs/SHREC17_R10_RI6_P6_PATCH10_SAMPLE5_FC_{level}")
        # generateMeshGraphDatasetFromPatches("Datasets/Patches/SHREC17_R10_RI6_P6", 2, 10, level, f"SHREC17_R10_RI6_P6_PATCH10_SAMPLE2_FC_{level}")
        # generateMeshGraphDatasetForNNTraining(f"Datasets/MeshGraphs/SHREC17_R10_RI6_P6_PATCH10_SAMPLE2_FC_{level}")
        # generateMeshGraphDatasetFromPatches("Datasets/Patches/SHREC17_R10_RI6_P6", 20, 25, level, f"SHREC17_R10_RI6_P6_PATCH25_SAMPLE20_FC_{level}")
        # generateMeshGraphDatasetForNNTraining(f"Datasets/MeshGraphs/SHREC17_R10_RI6_P6_PATCH25_SAMPLE20_FC_{level}")
        generateMeshGraphDatasetFromPatches("Datasets/Patches/SHREC17_R10_RI6_P6", 10, 50, level, f"SHREC17_R10_RI6_P6_PATCH50_SAMPLE10_FC_{level}")
        generateMeshGraphDatasetForNNTraining(f"Datasets/MeshGraphs/SHREC17_R10_RI6_P6_PATCH50_SAMPLE10_FC_{level}")


if __name__ == "__main__":
    main()
