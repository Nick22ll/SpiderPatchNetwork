from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import time
import json
import pickle
from itertools import product
from collections import namedtuple
from collections import OrderedDict


class RunManager:
    def __init__(self, path):
        self.path = path

        self.val_data = []
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        self.best_acc = 0
        self.best_loss = 100000
        self.best_results = OrderedDict()
        self.best_results["accuracy_params"] = None
        self.best_results["loss_params"] = None



        self.network = None
        self.loader = None
        self.tb = None

    def begin_run(self, run, network, loader):

        self.run_start_time = time.time()

        self.run_params = run
        self.run_count += 1

        self.network = network
        self.loader = loader

        self.tb = SummaryWriter(log_dir=f"{self.path}/TrainingRuns", comment=f'-{run}', flush_secs=60)

    def end_run(self):
        self.tb.close()
        self.epoch_count = 0

    def begin_epoch(self):
        self.epoch_start_time = time.time()

        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0

    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss / len(self.loader)
        accuracy = self.epoch_num_correct / len(self.loader.dataloader.dataset)

        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)

        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results['loss'] = loss
        results["accuracy"] = accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k, v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)

        return pd.DataFrame.from_dict(self.run_data, orient='columns')

    def track_loss(self, loss):
        self.epoch_loss += loss.item()

    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self.get_num_correct(preds, labels)

    def get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    def save(self):

        filename = f"training_run_{time.strftime('%a, %d %b %Y %H-%M', time.localtime(self.run_start_time))}"

        pd.DataFrame.from_dict(
            self.run_data, orient='columns'
        ).to_csv(f'{self.path}/TrainingResults/{filename}.csv')

        with open(f'{self.path}/TrainingResults/{filename}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)

    def track_validation(self, accuracy_score, loss, confusion_matrix=None):


        val_results = OrderedDict()
        val_results["epoch"] = self.epoch_count
        val_results["accuracy"] = accuracy_score
        val_results["loss"] = loss

        self.val_data.append(val_results)

        for k, v in self.run_params._asdict().items(): val_results[k] = v

        if accuracy_score > self.best_acc:
            self.best_acc = accuracy_score
            self.best_results["accuracy_params"] = val_results

        if loss < self.best_loss:
            self.best_loss = loss
            self.best_results["loss_params"] = val_results


        if confusion_matrix is not None:
            filename = f"validation"

            for k, v in self.run_params._asdict().items(): filename += f"_{k}{v}"

            filename += f"_{time.strftime('%d %b %Y %H-%M', time.localtime(self.run_start_time))} - Epoch{self.epoch_count}"

            with open(f"{self.path}/TrainingValidationConfusionMatrices/" + filename, 'wb') as f:
                pickle.dump(confusion_matrix, f)

    def save_validation(self):
        filename = f"validation"

        for k, v in self.run_params._asdict().items(): filename += f"_{k}{v}"

        filename += f"_{time.strftime('%d %b %Y %H-%M', time.localtime(self.run_start_time))}"

        pd.DataFrame.from_dict(
            self.val_data, orient='columns'
        ).to_csv(f'{self.path}/TrainingValidations/{filename}.csv')

        with open(f'{self.path}/TrainingValidations/{filename}.json', 'w', encoding='utf-8') as f:
            json.dump(self.val_data, f, ensure_ascii=False, indent=4)

        with open(f'{self.path}/TrainingValidationsBestResults/best_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.best_results, f, ensure_ascii=False, indent=4)

        self.val_data = []

    def disaply_dataframe(self, dataframe_list):
        with pd.option_context('display.max_rows', 20,
                               'display.max_columns', None,
                               'display.precision', 3,
                               'display.width', None,
                               'display.colheader_justify', 'center',
                               ):
            print("Top 20 accuracies!")
            print(pd.DataFrame.from_dict(dataframe_list).sort_values('accuracy', ascending=False))
            print("\n\nTop 20 loss!")
            print(pd.DataFrame.from_dict(dataframe_list).sort_values('loss', ascending=True))

class RunBuilder:
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs
