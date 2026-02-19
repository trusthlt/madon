import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../experiments-holistic-formalism-feature-based')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from utils.utilities import *
from process_features import ProcessFeatures
from configuration import Configuration
from train import Trainer
from significance import Significance

from data.processing.create_dataset import CreateDataset

def __main__():
    # set and get the parameters to shape the frame of the project
    data_path = '../../../data/dataset'
    ds_path = os.path.join(data_path, 'processed_datasets')

    parameters = get_parameters()
    train_val = parameters['train']

    # initialize the configuration at backend (choice of device, fixing the seed)
    configuration = Configuration(parameters)

    dataset_obj = CreateDataset(ds_path, parameters['is_gold'], parameters['split'])

    # process the features to create dataset for feature engineering
    proc_features = ProcessFeatures(dataset_obj, parameters=parameters)

    train_obj = Trainer(parameters, config_obj=configuration, feat_proc_obj=proc_features)
    train_obj.evaluate_best(split='test')

    parameters['train'] = train_val

    if parameters['train']:
        # if train was set true, then model will be trained (unless, the same experiment num was used before)
        train_obj.train()

    if parameters['test']:
        train_obj.evaluate_best(split='test', inference=False)

    significance_obj = Significance(train_obj, parameters, configuration, proc_features)
    significance_obj.explain_predictions('test')


if __name__ == '__main__':
    __main__()