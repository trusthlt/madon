import json
import os
import pickle

class ReadData:
    def __init__(self, result_dir: str , curated_data_path: str, raw_data_path: str, annotators: list):
        """
        Class is used to read the data from the given paths
        :param result_dir: Resulting directory where further processed data will be stored
        :param curated_data_path: file path for curated data
        :param raw_data_path: file path for raw data
        :param annotators: list of annotators, which is required to extract the annotations (for scoring)
        """
        self.curated_path = curated_data_path
        self.raw_path = raw_data_path
        self.check_dir(result_dir)
        self.annotators = annotators
        self.result_dir = result_dir
        self.curated_data = self.process()

    @staticmethod
    def check_dir(directory: str) -> None:
        """
        Method that checks if the given directory exists
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

    def get_curated_list(self) -> list:
        """
        Method is used to get the list of curated data files
        :return: list of curated data files
        """
        if not os.path.exists(self.curated_path):
            raise FileNotFoundError(f"Curated data path {self.curated_path} does not exist.")
        return os.listdir(self.curated_path)

    @staticmethod
    def load_data(path: str) -> dict:
        """
        Given a path, method is used to load the data from the document
        :param path: file path either for raw or curated data
        :return: dictionary which contains the un-processed data
        """
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def collect_curations(self) -> list:
        """
        Method is used to collect curated data, which will be helpful for further processing and direct access
        :return: list of datapoints by their file paths (curated, raw, annotations (will be used for scores))
        """
        curated_data = list()
        annotation_path = os.path.join(self.raw_path, 'annotation')
        for each in os.listdir(self.curated_path):
            if each == '.DS_Store':
                continue
            current_path = os.path.join(self.curated_path, each)
            files = [os.path.join(current_path, file) for file in os.listdir(current_path) if file.endswith('.json')]
            raw_folder = os.path.join(annotation_path, each)
            raw_file = os.path.join(raw_folder, f'INITIAL_CAS.json')

            curated_data.append(
                {
                    'data_id': each,
                    'curated_data':self.load_data(files[0]),
                    'raw_data': self.load_data(raw_file),
                    'annotated_data': {
                        annotation_file.replace('.json', ''): self.load_data(
                            os.path.join(raw_folder, annotation_file)
                        ) for annotation_file in os.listdir(raw_folder)
                        if 'INITIAL_CAS' not in annotation_file and annotation_file.replace('.json', '') in self.annotators
                    }
                }
            )
        return curated_data

    def process(self) -> list:
        """
        Method functions as a main method for the class. It helps us eliminate redundant processing time.
        :return: list of datapoints by their file paths (curated, raw, annotations (will be used for scores))
        """
        curated_dataset = os.path.join(self.result_dir, 'inception_data.pickle')

        if not os.path.exists(curated_dataset):
            curated_data = self.collect_curations()
            with open(curated_dataset, 'wb') as f:
                pickle.dump(curated_data, f)
        with open(curated_dataset, 'rb') as f:
            curated_data = pickle.load(f)
        return curated_data

    def __len__(self) -> int:
        """
        Method is used to get the length of the data was extracted from metadata files
        :return: number of datapoints
        """
        return len(self.curated_data)

    def __getitem__(self, idx: int) -> dict:
        """
        Method is used to get the data point by its index
        :param idx: integer specifies the index of the data point
        :return: specified data point
        """
        if idx >= self.__len__():
            raise StopIteration
        return self.curated_data[idx]
