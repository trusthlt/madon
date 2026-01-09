import pickle
from ast import literal_eval
import os
import pandas as pd

# from data.processing.b_process_raw_data import ProcessData


class CreateDataset:
    def __init__(self, ds_path: str, is_gold: bool=True, split: bool=False):
        """
        initializer of the dataset creator object
        :param ds_path: Dataset path
        :param is_meta: specifies whether the meta (raw) data is created (True) or gold standard (False)
        :param split: specifies whether data will be split (True) or not (False)
        """
        self.ds_path = ds_path
        self.ds_name = None
        self.is_gold = is_gold
        self.split = split
        # self.ds_idx = 0 if self.is_meta else 1
        self.experimental_label_map = self.set_exp_label_map()
        self.raw_dataset = self.get_raw_data()
        self.gold_label_map = self.raw_dataset["labels"]
        self.dataset_type = 'gold' if self.is_gold else 'fine-grained'
        self.result_path = os.path.join(self.ds_path, self.dataset_type)
        self.check_dir(self.result_path)
        self.dataset = self.__main__()

    def get_raw_data(self):
        meta_path = os.path.join(self.ds_path, 'descriptive-raw-dataset.pickle')
        return pickle.load(open(meta_path, 'rb'))

    @staticmethod
    def check_dir(directory: str) -> None:
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def check_paragraph_labels(paragraph: dict, label_info: list) -> tuple:
        """
        Method is used to collect labels per paragraphs. Notice that there are cases that paragraphs do not have labels
        :param paragraph: Dictionary that contains paragraph information
        :param label_info: label information per document that is used to align with a paragraph
        :return: tuple, which contains labels and argument information
        """
        labels = list()
        argument_info = list()
        result_label = list()
        result_info = list()

        for label in label_info:
            if label['begin'] <= paragraph['begin'] <= label['end'] or label['begin'] <= paragraph['end'] <= label['end']:
                label_info = {
                    'begin': label['begin'],
                    'end': label['end'],
                    'tokens': label['tokens']
                }
                if label['label'] == 'RRESULT':
                    result_label.append(label['label_ui'])
                    result_info.append(label_info)
                    continue
                labels.append(label['label'])
                argument_info.append(label_info)

        return labels, argument_info, result_label, result_info

    @staticmethod
    def tokenize_paragraph(paragraph: dict, tokens: dict) -> list:
        """
        Given paragraph and token information, each paragraph is tokenized using begin and end indexes
        :param paragraph: dictionary that contains paragraph relevant information from INCePTION
        :param tokens: dictionary which contains token relevant information from INCePTION
        :return: list of tokens per paragraph, which is tokenized using INCePTION tokens
        """
        tokenized = [
            token['text'] for token in tokens
            if paragraph['begin'] <= token['begin'] and token['end'] <= paragraph['end']
        ]

        return tokenized

    @staticmethod
    def set_exp_label_map() -> dict:
        """
        Method is used to generate the dictionary to map gold labels into experimental labels
        :return: mapping dictionary
        """
        experimental_label_map = {
            "PCPRACTICALCONSEQUENCES": "PC",
            "INTCLCASELAW": "CL",
            "INTDDOCTRINE": "D",
            "EXTHISTORICALINTERPREATION": "HI",
            "EXTTELEOLOGICALINTERPRETATION": "TI",
            "TLATRADITIONALLEGALARGUMENTS_RLA": "TI",
            "LILINGUISTICINTERPRETATION": "LIN",
            "TLATRADITIONALLEGALARGUMENTS_AAC": "LIN",
            "SISISTEMICINTERPRETATION": "SI",
            "EUEUCIEULAWCONFORMINGINTERPRETATION": "SI",
            "CONSTCCICONSTITUTIONALCONFORMINGINTERPRETATION": "SI",
            "EXTPLPRINCIPLESOFLAW": "PL",
            "CONSTCVCONSTITUTIONALVALUESANDPRINCIPLES": "PL",
            "EUEUVEUVALUESANDPRINCIPLES": "PL",
            "RRESULT": "RRESULT",
        }
        return experimental_label_map

    def update_labels(self, golds: list, arg_infos: list) -> dict:
        """
        Method is used to update the labels of each document
        :param golds: list of gold labels for the given input
        :param arg_infos: metadata for the given input, which includes all relevant information for argument
        :return: dictionary which contains argument labels and argument information [begin, end, tokens]
        """

        info_dict = {
            'labels': list(),
            'arguments': list()
        }
        if golds:
            for label, args in zip(golds, arg_infos):
                if label in self.experimental_label_map.keys():
                    exp_label = self.experimental_label_map[label]
                    info_dict['labels'].append(exp_label)
                    info_dict['arguments'].append(args)

        return info_dict


    def get_paragraph_info(self, document_id: int, document_info: dict) -> dict:
        """
        Method is used to get paragraph level information
        :param document_id: document index in the standardized form (not the original document name)
        :param document_info: all information from the metadata of the document
        :return: dictionary which contains paragraph relevant information
        """
        document_paragraphs = {
            'paragraph_id': list(),
            'doc_id': list(),  # since it can be used for document mapping, we keep it as integer
            'text': list(),
            'labels': list(),
            'tokenized_text': list(),
            'f_labels': list(),
            'argument_info': list(),
            'result_info': list(),
            'result_labels': list()
        }

        for idx, paragraph in enumerate(document_info['paragraphs']):

            labels, argument_info, result_label, result_info = self.check_paragraph_labels(paragraph, document_info['argument_labels'])

            tokenized = self.tokenize_paragraph(paragraph, document_info['tokens'])
            document_paragraphs['paragraph_id'].append(f'doc_{document_id}_par_{idx}')
            document_paragraphs['doc_id'].append(document_id)
            document_paragraphs['text'].append(paragraph['text'])
            document_paragraphs['labels'].append(labels)
            document_paragraphs['tokenized_text'].append(tokenized)
            document_paragraphs['f_labels'].append(document_info['f_labels'][0]['label_ui'])
            document_paragraphs['argument_info'].append(argument_info)
            document_paragraphs['result_labels'].append(result_label)
            document_paragraphs['result_info'].append(result_info)

        return document_paragraphs

    def create_informative_dataset(self) -> dict:
        """
        Method is used to create the gold dataset (not the experimental dataset)
        :return: dictionary for the gold dataset
        """

        ds_path = os.path.join(self.result_path, f'dataset-access-by-id.pickle')
        gold_dataset = dict()
        if not os.path.exists(ds_path):
            for doc_id, doc in self.raw_dataset['dataset'].items():
                gold_dataset[doc_id] = self.get_paragraph_info(doc_id, doc)
            with open(ds_path, 'wb') as gold_file:
                pickle.dump(gold_dataset, gold_file)

        with open(ds_path, 'rb') as gold_file:
            gold_dataset = pickle.load(gold_file)

        return gold_dataset

    def __main__(self) -> pd.DataFrame:
        """
        Method is used as a __main__ function (as it's seen from the name) of the class
        :return: Dataframe of the dataset, where each datapoint is a paragraph
        """
        dataset = {
            'doc_id': list(), 'doc_name': list(), 'text': list(), 'labels': list(), 'f_labels': list(), 'result_labels': list(),
            'argument_info': list(), 'tokenized_text': list()
        }

        self.ds_name = f'dataset-{self.dataset_type}-all'
        dataset_path = os.path.join(self.result_path, self.ds_name)
        dataset_descriptive_path = os.path.join(self.result_path, f'descriptive-{self.ds_name}')

        map_reverse = {v: k for k, v in self.raw_dataset['doc_map'].items()}
        if not os.path.exists(dataset_path):
            informative_set = self.create_informative_dataset()
            for doc_id, doc_details in informative_set.items():
                dataset['doc_id'].append(doc_id)
                dataset['doc_name'].append(map_reverse[doc_id])

                labels = list()
                arguments = list()
                for label, argument in zip(doc_details['labels'], doc_details['argument_info']):
                    if self.is_gold:
                        # something to be discussed later
                        current_info = self.update_labels(label, argument)
                        labels.append(current_info['labels'])
                        arguments.append(current_info['arguments'])
                    else:
                        labels.append(label)
                        arguments.append(argument)

                dataset['labels'].append(labels)
                dataset['argument_info'].append(arguments)
                dataset['tokenized_text'].append(doc_details['tokenized_text'])

                dataset['text'].append(doc_details['text'])
                dataset['f_labels'].append(doc_details['f_labels'][0])
                dataset['result_labels'].append(doc_details['result_labels'])

            ds_df = pd.DataFrame(dataset)
            ds_df.to_csv(dataset_descriptive_path + '.csv', index=False)
            ds_df.drop(columns=['argument_info', 'tokenized_text']).to_csv(dataset_path + '.csv', index=False)

        ds_df = pd.read_csv(dataset_path + '.csv', converters={'labels': literal_eval, 'result_labels': literal_eval, 'argument_info': literal_eval})

        dataset = self.split_dataset(ds_df, self.result_path) if self.split else ds_df

        return dataset


    def split_dataset(self, df: pd.DataFrame, ds_path: str) -> dict:
        """
        Method is used to split the dataset into train, dev and test sets (by preserving representation of the courts
        and holistic labels as similar as possible)
        :param df: dataframe for the dataset
        :param ds_path: path where to save the dataset
        :return: dictionary of the train, dev and test sets
        """
        check_path = os.path.join(ds_path, f'train.csv') # all will be prepared together. If one does not exist, then all don't too.
        result_dict = {'train': None, 'dev': None, 'test': None}
        if not os.path.exists(check_path):
            doc_dict = {
                'SC': list(), 'ASC': list()
            }

            for doc_name, idx in self.raw_dataset['doc_map'].items():
                key = 'SC' if 'ECLI' in doc_name else 'ASC'
                doc_dict[key].append(idx)


            label_court_dict = dict()
            for k, lists in doc_dict.items():
                current_dict = {
                    'Formalistic': list(), 'Non-Formalistic': list()
                }
                for idx in lists:

                    key = 'Non-Formalistic' if 'NON' in list(df[df['doc_id'] == idx]['f_labels'])[0] else 'Formalistic'
                    current_dict[key].append(idx)
                label_court_dict[k] = current_dict

            split_guide = {'train': 0.7, 'dev': 0.2, 'test': 0.1}
            result_dict = {'train': list(), 'dev': list(), 'test': list()}
            for k, dicts in label_court_dict.items():
                for label, lists in dicts.items():

                    init_idx = 0
                    for k, split_ratio in split_guide.items():
                        end_idx = init_idx + round(len(lists) * split_ratio) if k != 'test' else len(lists)
                        result_dict[k].extend(lists[init_idx: end_idx])
                        init_idx = end_idx

            for k, idx_list in result_dict.items():
                current_dataset = df[df['doc_id'].isin(idx_list)]
                # input(current_dataset)
                ## last update, we use doc names as ids from now on
                current_dataset.drop(columns=['doc_id'], inplace=True)
                current_dataset.rename(columns={'doc_name': 'doc_id'}, inplace=True)
                file_path = os.path.join(ds_path, f'{k}.csv')
                current_dataset.to_csv(file_path, index=False)
        for k in result_dict.keys():
            file_path = os.path.join(ds_path, f'{k}.csv')
            result_dict[k] = pd.read_csv(file_path, converters={'labels': literal_eval, 'result_labels': literal_eval, 'argument_info': literal_eval})
        return result_dict

    @staticmethod
    def transform_dataset(list_data: list, ds_path: str) -> pd.DataFrame:
        """
        Method is used to transform the list of dataset into a DataFrame object.
        :param list_data: list of paragraphs, where each paragraph has several parameters to be considered
        :param ds_path: path where the dataframe will be saved after splitting operation is done
        :return: DataFrame object for the dataset
        """
        dataset = {k: list() for k in list_data[0]}
        for each in list_data:
            for k, val in each.items():
                dataset[k].append(val)
        df = pd.DataFrame(dataset)
        df.to_csv(ds_path.replace('.pickle', '.csv'))
        return df
