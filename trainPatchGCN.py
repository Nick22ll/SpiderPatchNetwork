from SpiderPatch.RunUtils import *

from PlotUtils import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
from SpiderPatch.Networks import *

from dgl.dataloading import GraphDataLoader

from SpiderDatasets.PatchDataset import PatchDatasetForNNTraining


def main():

    # Scelgo le feature con cui fare l'addestramento
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = PatchDatasetForNNTraining()
    dataset.loadFromRawPatchesDataset(10, 6, 6)
    dataset.to(device)
    #dataset.save_to("Datasets/MeshGraphs/SHREC17_R10_RI6_P6")

    in_feats_dim = dataset.aggregateNodeFeatures()
    dataset.aggregateEdgeFeatures()
    model = PatchConv2LayerClassifier(in_feats=in_feats_dim[1], hidden_dim=150, out_feats=dataset.train_dataset.numClasses()).to(device)
    train(model, dataset, batch_size=500, epochs=1000, lr=0.001)
    #model.save("TrainedModels/SHREC17_GCN2_18_6_8")
    #model.load("TrainedModels/SHREC17_GCN2")
    acc, loss, cm = test(model, dataset.validation_dataset)
    plot_confusion_matrix(cm)
    print(f"Accuracy score: {acc}\nLoss score: {loss}")


def greedySearch(dataset, params=None, device=None):
    if params is None:
        params = OrderedDict(
            lr=[.01]
            , batch_size=[8, 16, 64]
            , hidden_dim=[100, 250, 500]
        )

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_path = f"GreedySearchResults/{dataset.name}"
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(f"{base_path}/TrainingResults", exist_ok=True)
    os.makedirs(f"{base_path}/TrainingRuns", exist_ok=True)
    os.makedirs(f"{base_path}/TrainingValidations", exist_ok=True)
    os.makedirs(f"{base_path}/TrainingValidationConfusionMatrices", exist_ok=True)
    os.makedirs(f"{base_path}/TrainingValidationsBestResults", exist_ok=True)

    in_feats_dim = dataset.aggregateNodeFeatures()
    dataset.aggregateEdgeFeatures()

    feature_names = "aggregated_feats"
    edge_feats_name = "weights"

    run_manager = RunManager(base_path)
    run_counter = 1
    for run in RunBuilder.get_runs(params):
        print(f"Starting run: {run_counter}  |  lr =  {run.lr}  |  batch_size =  {run.batch_size}  |  hidden_dim =  {run.hidden_dim}")

        network = PatchConv2LayerClassifier(in_feats=in_feats_dim[1], hidden_dim=run.hidden_dim, out_feats=dataset.train_dataset.numClasses()).to(device)
        dataloader = GraphDataLoader(dataset.train_dataset, batch_size=run.batch_size, drop_last=False)
        optimizer = torch.optim.Adam(network.parameters(), lr=run.lr)

        run_manager.begin_run(run, network, dataloader)
        for epoch in range(1, 51):
            print(f"Epoch: {epoch}")
            run_manager.begin_epoch()
            for batched_graph, labels in dataloader:
                feats = batched_graph.ndata[feature_names]
                edge_weights = batched_graph.edata[edge_feats_name]
                pred = network(batched_graph, feats, edge_weights)
                loss = F.cross_entropy(pred, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                run_manager.track_loss(loss)
                run_manager.track_num_correct(pred, labels)
            run_manager.end_epoch()
            if epoch % 10 == 0 and epoch != 0:
                val_acc, val_loss, cm = test(network, dataset.validation_dataset, batch_size=run.batch_size, device=device)
                run_manager.track_validation(val_acc, val_loss, cm)
        run_manager.save_validation()
        run_manager.end_run()
        run_counter += 1
    run_manager.save()


def train(model, dataset, feature_names="aggregated_feats", edge_feats_name="weights", batch_size=50, epochs=1000, lr=0.01):
    # dataloader = GraphDataLoader(dataset.train_dataset, batch_size=batch_size, drop_last=False)
    # opt = torch.optim.Adam(model.parameters(), lr=lr)
    # for epoch in range(epochs):
    #     print(f"Epoch: {epoch}")
    #     for batched_graph, labels in dataloader:
    #         feats = batched_graph.ndata[feature_names]
    #         edge_weights = batched_graph.edata[edge_feats_name]
    #         pred = model(batched_graph, feats, edge_weights)
    #         loss = F.cross_entropy(pred, labels)
    #         opt.zero_grad()
    #         loss.backward()
    #         opt.step()
    #
    #     results = test(model=model, dataset=dataset.validation_dataset, batch_size=batch_size)
    #     print(results[:2])

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        for i in range(len(dataset.train_dataset.graphs)):
            graph = dataset.train_dataset.graphs[i]
            label = dataset.train_dataset.labels[i]
            feats = graph.ndata[feature_names]
            edge_weights = graph.edata[edge_feats_name]
            pred = model(graph, feats, edge_weights)
            loss = F.cross_entropy(pred, label)
            opt.zero_grad()
            loss.backward()
            opt.step()

        results = test(model=model, dataset=dataset.validation_dataset, batch_size=batch_size)
        print(results[:2])

def test(model, dataset, node_feat_names="aggregated_feats", edge_feats_name="weights", batch_size=20, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataloader = GraphDataLoader(dataset, batch_size=batch_size, drop_last=False)
    correct_prediction_number = torch.empty(0, device=device)
    loss_predicted = torch.empty(size=(0, dataset.numClasses()), device=device)
    for batched_graph, labels in test_dataloader:
        feats = batched_graph.ndata[node_feat_names]
        edge_weights = batched_graph.edata[edge_feats_name]
        pred = model(batched_graph, feats, edge_weights)
        correct_prediction_number = torch.hstack((correct_prediction_number, pred.argmax(dim=1)))  # Take the highest value in the predicted classes vector
        loss_predicted = torch.vstack((loss_predicted, pred))

    # Computation of accuracy metrics
    accuracy = correct_prediction_number.eq(dataset.labels).sum().item() / len(test_dataloader.dataloader.dataset)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        confusion_matrix = metrics.confusion_matrix(dataset.labels.tolist(), correct_prediction_number.tolist(), normalize="true")
    final_labels = torch.tensor(dataset.labels.tolist(), dtype=torch.int64, device=device)
    loss = F.cross_entropy(loss_predicted, final_labels)
    return accuracy, loss.item(), confusion_matrix


if __name__ == "__main__":
    main()
