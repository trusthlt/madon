import os
import pickle
from read_data import ReadData

class ProcessData:
    def __init__(self, reader: ReadData, result_path: str):
        """
        initializer for the raw data processing object
        :param reader:
        :param result_path:
        """
        self.reader = reader
        self.result_dir = result_path
        self.doc_data, self.map, self.labels = self.__main__()


    @staticmethod
    def get_text(datapoint: dict) -> str:
        """
        Method provides text information given the datapoint
        :param datapoint: dictionary for the datapoint
        :return:text information from the given datapoint
        """
        for each in datapoint['%FEATURE_STRUCTURES']:
            if 'Sofa' in each['%TYPE']:
                return each['sofaString']

    @staticmethod
    def get_information(data: dict, text: str) -> dict:
        """
        Method process to standardize the relevant information for the preprocessing
        :param data: dictionary for the specific part of the datapoint (paragraph or token)
        :param text: raw text to be processed according to the data information
        :return: dictionary to hold relevant information
        """

        return {
            'id': data['%ID'],
            'begin': data['begin'],
            'end': data['end'],
            'text': text[data['begin']:data['end']]
        }

    def collect_main_information(self, data: dict) -> tuple:
        """
        Method is used to collect main information from the given datapoint
        :param data: dictionary which contains raw data from INCePTION for the given document
        :return: tuple that contains list of tokens and list of paragraphs
        """
        tokens = list()
        paragraphs = list()
        text = self.get_text(data)

        for structure in data['%FEATURE_STRUCTURES']:
            if 'Token' in structure['%TYPE']:
                tokens.append(self.get_information(structure, text))
            elif 'Paragraph' in structure['%TYPE']:
                paragraphs.append(self.get_information(structure, text))
            else:
                continue
        return tokens, paragraphs

    @staticmethod
    def get_labels(data: dict) -> list:
        """
        Method is used to collect labels for the specific document
        :param data: dictionary which contains data features
        :return: list of labels per span, where label includes begin and end of the span and label information
        """
        unnecessary_info = ['%ID', '%TYPE', 'begin', 'end', '@sofa']

        labels = list()
        for k, v in data.items():
            if k not in unnecessary_info:
                label_ui = v
                label = k
                current_data = {
                    'begin': data['begin'],
                    'end': data['end'],
                    'label_ui': label_ui,
                    'label': label
                }
                labels.append(current_data)

        return labels

    @staticmethod
    def check_traditional_argument(label_list: list) -> list:
        """
        Method is used to collect subcategories of Traditional Legal Argument category. Specific case.
        :param label_list: list of labels that is extracted with the span information
        :return: new list in which TLA category is updated with its subcategories: RLA and AAC
        """
        new_list = list()
        for each in label_list:
            label_name = each['label']
            if "TLA" in each['label']:
                addition = 'AAC' if 'AAC' in each['label_ui'] else 'RLA'
                label_name += f"_{addition}"

            updated_data_point = each.copy()
            updated_data_point['label'] = label_name
            new_list.append(updated_data_point)
        return new_list

    def collect_argument_categories(self, data, doc_id=None) -> tuple:
        """
        Method is used to collect legal argument labels from the annotated data
        :param data: dictionary that contains curated data
        :param doc_id: document id that will be used in special scenarios (documents without arguments) will be checked
        :return: tuple that contains list of argument labels and list of formalistic labels
        """

        argument_labels = list()
        formalistic_labels = list()

        docs_with_non_args = list()

        for structure in data['%FEATURE_STRUCTURES']:
            if 'LegalArgument' in structure['%TYPE']:

                labels = self.get_labels(structure)

                if not labels: # for cases where the label is not present but highlighted
                    docs_with_non_args.append(doc_id)
                    break

                labels = self.check_traditional_argument(labels)

                if 'OVERALL' in labels[0]['label']: #expected overall is 1, if not then there is sth wrong!
                    formalistic_labels.extend(labels)
                else:
                    argument_labels.extend(labels)

            else:
                continue
        return argument_labels, formalistic_labels

    def tokenize_arguments(self, arguments: list, tokens: list) -> list:
        """
        Method is used to tokenize arguments using their token information from the inception
        :param arguments: legal arguments which include all relevant information for the "annotated" span
        :param tokens: list of tokens that are extracted per document using get_information(.) method
        :return: list of tokenized argument
        """
        arguments_tokenized = list()
        for argument in arguments:
            new_arg = argument.copy()
            tokenized = list()
            for token in tokens:
                if argument['begin'] <= token['begin'] and argument['end'] >= token['end']:
                    tokenized.append(token['text'])
                if token['end'] > argument['end']:
                    break
            new_arg['tokens'] = tokenized
            arguments_tokenized.append(new_arg)
        return arguments_tokenized

    def collect_annotations(self, data: dict) -> dict:
        """
        Method is used to collect annotations by (provided) annotators. This information is necessary for inter-coder
        agreement computation
        :param data: dictionary that contains annotated documents per annotator, where each document has the same
                     structure with the curated data
        :return: dictionary that contains legal argument and formalistic labels per annotator
        """
        annotation_dict = dict()
        for annotator, annotation in data.items():
            argument, formalistic = self.collect_argument_categories(annotation)
            annotation_dict[annotator] = {
                'argument_labels': argument,
                'formalistic_labels': formalistic
            }
        return annotation_dict



    def __main__(self):
        """
        Method is used as a main function of this object that performs following steps in order to collect required
        information by iterating through raw data (using reader object):
            1. checks if this data file exists (will be created if it does not -> useful for consistency);
            2. collects tokens and paragraphs with their required information (e.g., begin and end indexes);
            3. collects legal arguments (spans and labels) and formalistic labels;
            4. tokenizes the arguments that were collected as text using tokens from INCePTION;
            5. collect annotations per annotator to compute the inter-coder agreement for the formalistic labels;
            6. creates a dictionary for standard label mapping;
            7. combines all data in one dictionary:
                dataset: meta dataset which is developed based on raw data from INCePTION;
                doc_map: dictionary that contains mapping between original document name and index that we assign;
                labels: dictionary that contains mapping between labels and label_{idx}
            8. this dictionary is saved so it won't be re-created whenever code is run;
            9. if 1. returns False, then it loads already created dictionary;
        :return: tuple that contains all information that we mentioned at the step 7
        """
        document_map = dict()
        dataset = dict()
        labels = list()
        document_dataset = os.path.join(self.result_dir, f'descriptive-raw-dataset.pickle')
        if not os.path.exists(document_dataset):
            for idx, document in enumerate(self.reader):
                document_map[document['data_id']] = idx
                current_data = dict()

                current_data['tokens'], current_data['paragraphs'] = self.collect_main_information(document['raw_data'])
                arguments, current_data['f_labels'] = self.collect_argument_categories(document['curated_data'], document['data_id'])
                current_data['argument_labels'] = self.tokenize_arguments(arguments, current_data['tokens'])
                current_data['annotations'] = self.collect_annotations(document['annotated_data'])
                dataset[idx] = current_data # previously it was like this
                # dataset[document['data_id']] = current_data
                labels.extend([each['label'] for each in current_data['argument_labels']])

            result_set = {
                'dataset': dataset,
                'doc_map': document_map,
                'labels': list(set(labels))
            }
            with open(document_dataset, 'wb') as result_file:
                pickle.dump(result_set, result_file)
        with open(document_dataset, 'rb') as result_file:
            result_set = pickle.load(result_file)

        return result_set['dataset'], result_set['doc_map'], result_set['labels']

    def __len__(self) -> int:
        """
        Method to compute the length of the dataset
        :return: number of documents in the dataset
        """
        return len(self.doc_data)

    def __getitem__(self, item: int) -> dict:
        """
        Method is used to get specific item from the dataset
        :param item: index of the datapoint
        :return: specific datapoint
        """
        if item not in self.doc_data.keys():
            raise StopIteration
        return self.doc_data[item]
