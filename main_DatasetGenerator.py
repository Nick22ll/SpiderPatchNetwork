from SHREC_Utils import subdivide_for_mesh
from SpiderDatasets.MeshGraphDataset import MeshGraphDataset, MeshGraphDatasetForNNTraining
from SpiderPatch.Patch import *
from SpiderPatch.DatasetGenerator import *


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
    save_path = f"Datasets/Prova/{name[:name.rfind('_')].replace('_level', '')}/{new_name}"
    os.makedirs(save_path, exist_ok=True)
    dataset.save(save_path)


def main():
    # generatePatchDataset(start_idx=121)
    # for level in ["level_0", "level_1", "level_2", "level_3"]:  # , "all"
    #     generateMeshGraphDatasetFromPatches("Datasets/Patches/SHREC17_R10_RI6_P6", 10, 10, level, f"SHREC17_R10_RI6_P6_PATCH10_SAMPLE10_{level}")

    for level in ["level_0", "level_1", "level_2", "level_3"]:  # , "all"
        generateMeshGraphDatasetForNNTraining(f"Datasets/MeshGraphs/SHREC17_R10_RI6_P6_PATCH10_SAMPLE10_{level}")
    # generateMeshGraphDatasetForNNTraining(f"Datasets/MeshGraphs/SHREC17_R10_RI6_P6_PATCH10_{level}")

    # for folder in tqdm(os.listdir("Datasets/MeshGraphs"), position=0, leave=True, desc=f"Generating NNTraining Dataset", colour="green", ncols=120):
    #    generateMeshGraphDatasetForNNTraining("Datasets/MeshGraphs/" + folder)


if __name__ == "__main__":
    main()
