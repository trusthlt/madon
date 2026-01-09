import argparse
from argparse import Namespace
import os

def set_parameters() -> Namespace:
    parser = argparse.ArgumentParser()
    # Project Setup Parameters (whether to train or not, where to save data, how to navigate work)
    parser.add_argument('--statistics', required=False, action='store_true', default=False,
                        help='Collect data statistics or not')
    parser.add_argument('--data_path', required=False, type=str, default='../dataset',
                        help='Path to the directory where datasets are stored')
    parser.add_argument('--result_path', required=False, type=str, default='experiments',
                        help='Result path will be used to collect experiment results')

    # Dataset Setup
    parser.add_argument('--is_gold', required=False, action='store_true',
                        help='Specifies what kind of dataset should be created (True -> Gold, False -> Experimental)')
    parser.add_argument('--split', required=False, action='store_true',
                        help='Dataset should be split or not')

    # outliers
    parser.add_argument('--num_layers', required=False, type=int, default=3)
    parser.add_argument('--inference', action='store_true', default=False,)

    # process keys
    parser.add_argument('--read_data', action='store_true', required=False,
                        help='It is a key to call the raw data reader object and read the data (first process)')
    # parser.add_argument('--statistics', action='store_true', required=False,
    #                     help='It is a key to statistics object as collecting statistical information is optional')


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
