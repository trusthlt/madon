import argparse
from argparse import Namespace
import os

def list_of_dims(argument):
    return list(map(int, argument.split(',')))

def set_parameters() -> Namespace:
    parser = argparse.ArgumentParser()
    # Project Setup Parameters (whether to train or not, where to save data, how to navigate work)
    parser.add_argument('--train', required=False, action='store_true', default=False,
                        help='Activates the training session')
    parser.add_argument('--test', required=False, action='store_true', default=False,
                        help='Activates the testing session')
    parser.add_argument('--statistics', required=False, action='store_true', default=False,
                        help='Collect data statistics or not')
    parser.add_argument('--data_path', required=False, type=str, default='data',
                        help='Path to the directory where datasets are stored')
    parser.add_argument('--result_path', required=False, type=str, default='../task_3/MLP/experiments',
                        help='Result path will be used to collect experiment results')
    parser.add_argument('--exp_num', required=False, type=int, default=43,
                        help='Specifies experiment number')
    parser.add_argument('--show_results_only', required=False, action='store_true', default=False,
                        help='Bypasses training and shows saved results, if exists')
    parser.add_argument('--is_mac', required=False, action='store_true', default=False,
                        help='Defines whether the used device is mac or linux (when not set)')


    # Feature Engineering Setup
    parser.add_argument('--normalize', required=False, action='store_true', default=False,
                        help='Specifies whether features should be normalized or not')
    parser.add_argument('--epochs', required=False, type=int, default=10,
                        help='Number of epochs for mlp, number of max iterations for logistic regression')
    parser.add_argument('--learning_rate', required=False, type=float, default=1e-3,
                        help='Setting the learning rate for training phase')
    parser.add_argument('--batch_size', required=False, type=int, default=8,
                        help='Batch size for the mlp model, to be considered')
    parser.add_argument('--seed', required=False, type=int, default=32,
                        help='Random seed to fix the randomness')
    parser.add_argument('--optimizer', required=False, choices=['sgd', 'adam'], default='adam',
                        help='Optimizer to be used')
    parser.add_argument('--patience', required=False, type=int, default=3,
                        help='Patience for early stopping')
    parser.add_argument('--hidden_dim_list', nargs='+', default=[11, 20, 50],
                        help='List for the hidden dimensions. Notice the first element is input dimension.')
    parser.add_argument('--dropout_list', nargs='+', default=[0.4, 0.1, 0.0],
                        help='List for the dropouts. Notice the last element for the output, so no dropout.')
    parser.add_argument('--eval_all', required=False, action='store_true', default=False,
                        help='Helps us to evaluate all seed combinations for the inference setup')
    parser.add_argument('--inference_choice', required=False, type=str, choices=['finetune', 'finetune_filtered'],
                        help='Different scenarios are evaluated, and that is how we choose it')

    # Dataset Setup
    parser.add_argument('--is_gold', required=False, action='store_true',
                        help='Specifies what kind of dataset should be created (True -> Gold, False -> Experimental)')
    parser.add_argument('--split', required=False, action='store_true',
                        help='Dataset should be split or not')


    # outliers
    parser.add_argument('--is_par_level', required=False, action='store_true',
                        help='Dataset level: Either paragraph (True) or document (False)')
    parser.add_argument('--num_layers', required=False, type=int, default=3)
    parser.add_argument('--inference', action='store_true', default=False,)


    return parser.parse_args()

def get_parameters() -> dict:
    parameters = dict()
    params = set_parameters()
    for argument in vars(params):
        parameters[argument] = getattr(params, argument)
    return parameters

def check_dir(directory) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)
