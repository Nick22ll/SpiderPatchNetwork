import multiprocessing
import threading

from SpiderDatasets.MeshGraphDataset import MeshGraphDataset
from SpiderDatasets.MeshGraphForTrainingDataset import MeshGraphDatasetForNNTraining
from SpiderPatch.PatchDatasetGenerator import *
from CSIRS.Ring import Ring, ConcentricRings
import pickle as pkl


def generateMeshGraphDatasetFromPatches(patch_path, graph_per_mesh, patch_per_graph, resolution_level, neighbours_number, features_to_keep, dataset_name):
    dataset = MeshGraphDataset(dataset_name=dataset_name)
    dataset.fromRawPatches(patch_path, resolution_level=resolution_level, graph_for_mesh=graph_per_mesh, patch_for_graph=patch_per_graph, connection_number=neighbours_number, feature_to_keep=features_to_keep)
    dataset.save_to(f"Datasets/MeshGraphs/{dataset_name}")


def generateMeshGraphDatasetFromSuperPatches(patch_path, graph_per_mesh, patch_per_graph, neighbours_number, features_to_keep, dataset_name):
    dataset = MeshGraphDataset(dataset_name=dataset_name)
    dataset.fromRawSuperPatches(patch_path, graph_for_mesh=graph_per_mesh, patch_for_graph=patch_per_graph, connection_number=neighbours_number, feature_to_keep=features_to_keep)
    dataset.save_to(f"Datasets/MeshGraphs/Super_{dataset_name}")


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
    # mesh = Mesh()
    # mesh.load(f"U:\AssegnoDiRicerca\MeshDataset\SHREC17\PatternDB/609.off")
    # dataset = MeshGraphDataset()
    # dataset.load_from("Datasets/MeshGraphs/SHREC17_R10_RI6_P6_PATCH25_SAMPLE20_ARBITRARY_all", "SHREC17_R10_RI6_P6_PATCH25_SAMPLE20_ARBITRARY_all")
    # # with open("Datasets/Patches/SHREC17_R10_RI6_P6/0/level_0/0/patches609.pkl", "rb") as file:
    # #     ring = pkl.load(file)
    # #     mesh.draw_with_patches(ring)
    #     # for i in range(len(ring)):
    #     #     mesh.draw_with_patches([ring[i]])
    # #dataset.graphs[0].draw()
    # mesh.draw_with_MeshGraph(dataset.graphs[0])
    # mesh.drawWithLD(10)
    # mesh.drawWithGaussCurv(10)
    # mesh.drawWithMeanCurv(10)
    # mesh.drawWithK2(10)

    # thread_num = 6
    # pool = multiprocessing.Pool(processes=thread_num)
    # mesh_for_thread = int(180 / thread_num)
    # pool.map(parallelGenerateLRFPatchDataset, [r for r in [range(i*mesh_for_thread, (i*mesh_for_thread)+mesh_for_thread) for i in range(thread_num)]])

    #     # generateMeshGraphDatasetFromPatches("Datasets/Patches/SHREC17_R5_RI4_P6", 5, 25, level, f"SHREC17_R5_RI4_P6_PATCH25_SAMPLE5_FC_{level}")
    #     # generateMeshGraphDatasetForNNTraining(f"Datasets/MeshGraphs/SHREC17_R5_RI4_P6_PATCH25_SAMPLE5_FC_{level}")
    #     # generateMeshGraphDatasetFromPatches("Datasets/Patches/SHREC17_R7_RI4_P8", 10, 15, level, f"SHREC17_R7_RI4_P8_PATCH15_SAMPLE10_FC_{level}")
    #     # generateMeshGraphDatasetForNNTraining(f"Datasets/MeshGraphs/SHREC17_R7_RI4_P8_PATCH15_SAMPLE10_FC_{level}")
    #     # generateMeshGraphDatasetFromPatches("Datasets/Patches/SHREC17_R10_RI6_P6", 5, 10, level, f"SHREC17_R10_RI6_P6_PATCH10_SAMPLE5_FC_{level}")
    #     # generateMeshGraphDatasetForNNTraining(f"Datasets/MeshGraphs/SHREC17_R10_RI6_P6_PATCH10_SAMPLE5_FC_{level}")
    #     # generateMeshGraphDatasetFromPatches("Datasets/Patches/SHREC17_R10_RI6_P6", 2, 10, level, f"SHREC17_R10_RI6_P6_PATCH10_SAMPLE2_FC_{level}")
    #     # generateMeshGraphDatasetForNNTraining(f"Datasets/MeshGraphs/SHREC17_R10_RI6_P6_PATCH10_SAMPLE2_FC_{level}")
    #     # generateMeshGraphDatasetFromPatches("Datasets/Patches/SHREC17_R10_RI6_P6", 20, 25, level, f"SHREC17_R10_RI6_P6_PATCH25_SAMPLE20_FC_{level}")
    #     # generateMeshGraphDatasetForNNTraining(f"Datasets/MeshGraphs/SHREC17_R10_RI6_P6_PATCH25_SAMPLE20_FC_{level}")
    # generateMeshGraphDatasetFromSuperPatches("Datasets/SuperPatches/SHREC17_R10_RI6_P6", 20, 25, level, f"SHREC17_R10_RI6_P6_PATCH25_SAMPLE20_FC_{level}")
    # generateMeshGraphDatasetForNNTraining(f"Datasets/MeshGraphs/SHREC17_R10_RI6_P6_PATCH25_SAMPLE20_FC_{level}")
    generateMeshGraphDatasetFromPatches("Datasets/LRFPatches/SHREC17_R10_RI3_P4", 30, 20, "all", 0, features_to_keep=None, dataset_name=f"SHREC17_R10_RI3_P4_PATCH30_SAMPLE20_FC_LRF_all")
    generateMeshGraphDatasetForNNTraining(f"Datasets/MeshGraphs/SHREC17_R10_RI3_P4_PATCH30_SAMPLE20_FC_LRF_all")

if __name__ == "__main__":
    main()
