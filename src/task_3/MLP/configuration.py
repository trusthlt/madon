import torch.cuda
import os
import pickle
import random
import numpy as np

class Configuration:
    def __init__(self, parameters: dict):
        """
        initializer of the configuration object
        :param parameters: dictionary to hold the configuration parameters
        """
        self.parameters = parameters

        self.device = self.set_device()
        self.seed_everything()


    def set_device(self) -> str:
        """
        Method is utilized to set the device based on the is_mac parameter, since macs don't have cuda
        :return: cpu, cuda (nvidia) or mps (map silicone) as a computational device
        """
        if self.parameters['is_mac']:
            return 'mps' if torch.backends.mps.is_available() else torch.device('cpu')
        else:
            return 'cuda' if torch.cuda.is_available() else torch.device('cpu')

    def seed_everything(self) -> None:
        """
        In order to fix every process to the same seed, we initialize everything's seed here
        :return: None
        """
        print(f'seeding everything with new seed: {self.parameters["seed"]}')
        random.seed(self.parameters['seed'])
        os.environ['PYTHONHASHSEED'] = str(self.parameters['seed'])
        np.random.seed(self.parameters['seed'])
        torch.manual_seed(self.parameters['seed'])
        if self.parameters['is_mac']:
            torch.mps.manual_seed(self.parameters['seed'])
        else:
            torch.cuda.manual_seed(self.parameters['seed'])
            torch.cuda.manual_seed_all(self.parameters['seed'])
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def save_experimental_configuration(self, folder_path: str) -> None:
        """
        Method to save the experimental configuration
        :param folder_path:
        :return: None
        """
        config_file = os.path.join(folder_path, 'config.pickle')
        with open(config_file, 'wb') as config_data:
            pickle.dump(self.parameters, config_data)

    def load_experimental_configuration(self) -> None:
        """
        Method to load the experimental configuration for the given experiment number
        :return: None
        """
        current_parameters = self.parameters.copy()
        exp_folder = os.path.join(self.parameters['result_path'], f'exp_{self.parameters["exp_num"]}', f'seed_{self.parameters["seed"]}')
        param_path = os.path.join(exp_folder, 'config.pickle')

        if not os.path.exists(exp_folder):
            os.makedirs(exp_folder)
            with open(param_path, 'wb') as config_data:
                pickle.dump(self.parameters, config_data)
        with open(param_path, 'rb') as config_data:
            experimental_parameters = pickle.load(config_data)

        # When the model was trained, these parameters were not there. Thus, we skip them when we update
        # model parameters based on the loaded parameters. Because, we use the best experimental model
        # for the pipeline
        for k, v in experimental_parameters.items():
            if k not in ['inference_choice', 'eval_all', 'result_path']:
                current_parameters[k] = v

        return current_parameters