import pandas as pd
import torch
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from configuration import Configuration
from process_features import ProcessFeatures
from model import FeatModel
from dataset import FeatSet

class Trainer:
    def __init__(self, parameters: dict, config_obj: Configuration, feat_proc_obj: ProcessFeatures):
        """
        initializer for the trainer object
        :param parameters: configuration parameters for the specific setup
        :param config_obj: configuration object that provides specific setup
        :param feat_proc_obj: feature processor object that provides processed features dataset
        """
        self.parameters = parameters
        self.config_obj = config_obj
        self.feat_proc_obj = feat_proc_obj
        self.loss_fn = self.set_loss()
        self.model = self.set_model()
        self.optimizer = self.set_optimizer()

    @staticmethod
    def set_loss() -> CrossEntropyLoss:
        """
        method to set loss function
        :return: Cross Entropy Loss
        """
        return CrossEntropyLoss()

    def set_model(self) -> FeatModel:
        """
        Method to set the model (more of a dynamic setup)
        :return: Classifier model
        """
        hyperparams  = {
            'hidden_dims': [int(each) for each in self.parameters['hidden_dim_list']],
            'dropout': [float(each) for each in self.parameters['dropout_list']],
            'out_dim': 2
        }
        return FeatModel(hp=hyperparams).to(self.config_obj.device)

    def set_optimizer(self):
        """
        Method to set optimizer function
        :return: Either adam or SGD
        """
        if self.parameters['optimizer'] == 'adam':
            return torch.optim.Adam(params=self.model.parameters(), lr=self.parameters['learning_rate'])
        else:
            return torch.optim.SGD(params=self.model.parameters(), lr=self.parameters['learning_rate'])

    def data_load(self, split: str, shuffle: bool=True, inference: bool=False, seed_info: int=42) -> DataLoader:
        """
        Method is used to load the data in batches
        :param split: string to specify which dataset to load (train, val, test)
        :param shuffle: whether to shuffle the data
        :return: DataLoader object
        """
        ds = FeatSet(self.feat_proc_obj, split, self.config_obj, normalize=self.parameters['normalize'], inference=inference, seed_info=seed_info)
        return DataLoader(ds, batch_size=self.parameters['batch_size'], shuffle=shuffle)

    @staticmethod
    def compute_acc(predictions: torch.Tensor, labels: torch.Tensor) -> tuple:
        """
        Method computes the accuracy of the predictions
        :param predictions: prediction tensor
        :param labels: ground truth tensor
        :return: tuple of output (list), labels (list), accuracy (int) -> Number of accurate predictions in the batch
        """

        output = torch.argmax(predictions, dim=1).tolist()
        labels = labels.tolist()
        accuracy = [p == t for p, t in zip(output, labels)]
        return output, labels, sum(accuracy)

    def train(self) -> tuple:
        """
        Method is used to train the model and collect the results
        Steps:
            loads the data
            run through the training batches
            collect the results per epoch and save them
            when everything is done, get the best results

        Note: There is also patience check to set the early stopping criteria

        :return: tuple of best f1 score, and the epoch according to the training history
        """
        experiment_path = os.path.join(self.parameters['result_path'], f'exp_{self.parameters["exp_num"]}', f'seed_{self.parameters["seed"]}')
        self.check_dir(experiment_path)
        min_loss = 100
        if not 'model_checkpoints' in os.listdir(experiment_path):
            process_info = dict()
            patience = 0
            for epoch in range(self.parameters['epochs']):
                self.model.train()
                train_loader = self.data_load('train')
                num_batches = len(train_loader)
                epoch_loss = 0
                for batch in train_loader:
                    self.optimizer.zero_grad()
                    train_data = batch['data'].to(self.config_obj.device)
                    train_labels = batch['label'].to(self.config_obj.device)
                    output = self.model(train_data)
                    loss = self.loss_fn(output, train_labels)
                    epoch_loss += loss.item()
                    loss.backward()
                    self.optimizer.step()
                train_loss = epoch_loss / num_batches
                dev_loss, dev_acc, dev_f1, _, _ = self.evaluation(split='dev', shuffle_val=True)

                epoch_info = {
                    'epoch': int(epoch), 'dev_loss': round(dev_loss, 4), 'train_loss': round(train_loss, 4),
                    'dev_acc': round(dev_acc, 4), 'f1': round(dev_f1, 4)
                }
                for k, v in epoch_info.items():
                    print(f'{k}: {v}')
                print('<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>')

                if not process_info:
                    process_info = {k: [v] for k, v in epoch_info.items()}
                    continue
                for detail, value in epoch_info.items():
                    process_info[detail].append(value)
                self.save_model(experiment_path, epoch)
                if dev_loss < min_loss:
                    min_loss = dev_loss
                    patience = 0
                else:
                    patience += 1
                if patience >= self.parameters['patience']:
                    break
            self.save_details(experiment_path, process_info)
        else:
            print('INFO: Training is already done')
        best_f1, _, best_ep = self.get_best(experiment_path)

        return best_f1, best_ep

    @staticmethod
    def check_dir(directory: str) -> None:
        """
        Method to check if a directory exists
        :param directory: path to check
        :return: None
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

    def save_model(self, experiment_path: str, epoch: int) -> None:
        """
        Method to save the model
        :param experiment_path: specific experiment path that was already defined
        :param epoch: epoch information to save the model accordingly
        :return: None
        """
        ckpt_path = os.path.join(experiment_path, 'model_checkpoints')
        self.check_dir(ckpt_path)
        model_path = os.path.join(ckpt_path, f'model_{epoch}.pkl')
        torch.save(self.model.state_dict(), model_path)


    @staticmethod
    def save_details(experiment_path: str, process_info: dict) -> None:
        """
        Method to save the details of the model
        :param experiment_path: Specific experiment path that was already defined
        :param process_info: Epoch based results, which are collected during training
        :return: None
        """
        results_path = os.path.join(experiment_path, 'results.pickle')
        with open(results_path, 'wb') as f:
            pickle.dump(process_info, f)


    def evaluation(self, split: str, shuffle_val:bool, inference: bool=False, seed_info: int=42) -> tuple:
        """
        Method to evaluate the model
        :param split: either dev or test
        :param shuffle_val: whether to shuffle the data
        :return: tuple:
            eval_avg_loss - average loss for the evaluation,
            f1_micro, f1_macro - micro and macro f1 scores
        """
        dataloader = self.data_load(split, shuffle=shuffle_val, inference=inference, seed_info=seed_info)
        num_batches = len(dataloader)
        self.model.eval()
        eval_loss = 0
        predictions = list()
        targets = list()

        with torch.no_grad():
            for batch in dataloader:
                data = batch['data'].to(self.config_obj.device)
                labels = batch['label'].to(self.config_obj.device)
                output = self.model(data)
                predictions.extend(torch.argmax(output, dim=1).tolist())
                targets.extend(labels.tolist())
                loss = self.loss_fn(output, labels)
                eval_loss += loss.item()
            if split == 'test':
                self.save_predictions(predictions, targets, inference=inference)

            eval_avg_loss = eval_loss / num_batches
            f1_micro = f1_score(targets, predictions, average='micro')
            f1_macro = f1_score(targets, predictions, average='macro')
            prec = precision_score(targets, predictions, average='macro')
            recall = recall_score(targets, predictions, average='macro')
            cm = confusion_matrix(targets, predictions)
            if split == 'test':
                print(prec, recall, cm)
        return eval_avg_loss, f1_micro, f1_macro, prec, recall

    def save_predictions(self, predictions: list, targets: list, inference=False) -> None:
        inference_info = f'_inference_{self.parameters["inference_choice"]}' if inference else ''
        experiment_path = os.path.join(self.parameters['result_path'], f'exp_{self.parameters["exp_num"]}', f'seed_{self.parameters["seed"]}')
        test_predictions_path = os.path.join(experiment_path, f'predictions{inference_info}_fp.csv')

        test_data = self.feat_proc_obj.features['test']
        result = {
            'doc_id': test_data['doc_id'],
            'Gold Labels': targets,
            'Predicted Labels': predictions,
        }
        pd.DataFrame(result).to_csv(test_predictions_path, index=False)


    def get_best(self, experiment_path: str) -> tuple:
        """
        Method to get the best model
        :param experiment_path: specific experiment path that was already defined
        :return: tuple of the best f1 score, minimum loss, and the epoch which the first two occurred
        """
        with open(os.path.join(experiment_path, 'results.pickle'), 'rb') as f:
            process_info = pickle.load(f)
        best_f1 = max(process_info['f1'])
        min_loss = 100
        best_epoch = 0
        for idx, each in enumerate(process_info['f1']):
            if each == best_f1:
                min_loss = min(process_info['dev_loss'][idx], min_loss)
                best_epoch = process_info['epoch'][idx]

        self.plot_fig(process_info, experiment_path, best_f1)
        return best_f1, min_loss, best_epoch

    @staticmethod
    def plot_fig(saved_dict: dict, experiment_path: str, best_f1: float) -> None:
        """
        Method to plot the figure
        :param saved_dict: dictionary of the training results
        :param experiment_path: specific experiment path that was already defined
        :param best_f1: the best f1 score to be used in the title of the figure
        :return: None
        """
        figure, axis = plt.subplots(figsize=(15, 10))

        axis.set_title(f'Loss with F1 max {best_f1}')
        axis.plot(saved_dict['epoch'], saved_dict[f'train_loss'], label='train')
        axis.plot(saved_dict['epoch'], saved_dict[f'dev_loss'], label='dev')
        axis.plot(saved_dict['epoch'], saved_dict['f1'], label='f1')
        axis.set_xlabel('Epochs')
        axis.set_ylabel(f'Loss')
        axis.legend()
        figure.savefig(f'{experiment_path}/loss.png')
        plt.close('all')

    def evaluate_best(self, split: str, inference: bool=False, seed_info: int=42) -> tuple:
        """
        Method is used to evaluate the best model. Given the split name, and already defined parameters setup, the best
        model is chosen and evaluation is run accordingly
        :param split: string to specify which dataset to use
        :param inference: if argument_mining results must be used in evaluation
        :return: None
        """

        experiment_path = os.path.join(
            self.parameters['result_path'], f'exp_{self.parameters["exp_num"]}', f'seed_{seed_info}'
        )
        ckpt_path = os.path.join(experiment_path, 'model_checkpoints')
        best_f1, min_loss, best_epoch = self.get_best(experiment_path)
        model_path = os.path.join(ckpt_path, f'model_{best_epoch}.pkl')
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.to(self.config_obj.device)
        shuffle_val = False if split=='test' else True

        self.config_obj.seed_everything()
        dev_loss, dev_acc, dev_f1, precision, recall = self.evaluation(
            split, shuffle_val=shuffle_val, inference=inference, seed_info=seed_info
        )

        print(f'Dev loss: {dev_loss}, Dev acc: {dev_acc}, Dev f1: {dev_f1}')
        return dev_loss, dev_acc, dev_f1, precision, recall

    def evaluate_all_scenarios(self, split: str, inference: bool=False):
        if not self.parameters['eval_all']:
            loss_val, acc_val, f1_val, precision, recall = self.evaluate_best(split=split, inference=inference, seed_info=self.parameters['seed'])
            return loss_val, acc_val, f1_val, precision, recall
        generic_experiment_path = os.path.join(self.parameters['result_path'], f'exp_{self.parameters["exp_num"]}')
        f_path = os.path.join(generic_experiment_path, f'{self.parameters["inference_choice"]}_{self.parameters["seed"]}_inference_results_fp.csv')

        if not os.path.exists(f_path):
            results_dict = {'seeds': [32, 42, 52], 'precision': list(), 'recall': list(), 'f1': list(), 'accuracy': list()}

            for feature_seed in [32, 42, 52]:

                self.parameters['seed'] = feature_seed
                loss_val, acc_val, f1_val, precision, recall = self.evaluate_best(split=split, inference=inference, seed_info=feature_seed)
                results_dict['precision'].append(precision)
                results_dict['recall'].append(recall)
                results_dict['f1'].append(f1_val)
                results_dict['accuracy'].append(acc_val)

            results_df = pd.DataFrame(results_dict)
            results_df.to_csv(f_path, index=False)
            print(results_df)