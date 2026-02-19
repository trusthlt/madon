import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt

from spacy.tokens.doc import Counter


class DataStats:
    def __init__(self, reader_obj, process_data_obj, dataset_obj, is_gold, stats_folder):
        self.reader_obj = reader_obj
        self.process_data_obj = process_data_obj
        self.dataset_obj = dataset_obj
        self.is_gold = is_gold
        self.check_dir(stats_folder)
        self.stats_folder = stats_folder

    @staticmethod
    def check_dir(directory: str) -> None:
        if not os.path.exists(directory):
            os.makedirs(directory)

    def get_experimental_arguments(self, counter_data: dict) -> dict:
        """
        Method is used to collect experimental arguments given the counter dictionary of gold labels (annotation was
        done according to those labels)
        :param counter_data: dictionary for the counter of the gold labels
        :return: dictionary which contains experimental labels and their counts
        """
        exp_counter = dict()
        for label, value in counter_data.items():
            if label in self.dataset_obj.experimental_label_map.keys():
                label_name = self.dataset_obj.experimental_label_map[label]
                if label_name not in exp_counter:
                    exp_counter[label_name] = value
                else:
                    exp_counter[label_name] += value
        return exp_counter

    @staticmethod
    def clean_result(label: str) -> str:
        """
        Method is used to extract additional information (i.e., type) from the result label
        :param label: a string that carries information regarding the type of the result
        :return: extracted extra information from the given label string
        """
        label_new = label.replace('R - RESULT - ', '')
        label_type = re.sub(r' \([^)]*\)', '', label_new).lower()
        return label_type

    def get_result_type(self, arguments: list) -> str:
        """
        Method is used to extract RESULT type. This label is judge's decision rather than an argument. Thus, we extract
        its type
        :param arguments: list of dictionaries, which contains all arguments that were annotated
        :return: either clean version of result (if exists) else string of 'no result'
        """
        for each in arguments:
            if each['label'] == 'RRESULT':
                return self.clean_result(each['label_ui'])
        return 'no result'

    def get_args_per_doc(self) -> pd.DataFrame:
        """
        Method is used to generate the table for argument counts per document
        :return: dataframe for the aforementioned table
        """
        file_extension = 'meta' if not self.is_gold else 'gold'

        arg_stats_file = os.path.join(self.stats_folder, f'arg_stats_per_doc_{file_extension}.csv')
        if not os.path.exists(arg_stats_file):
            doc_stats = {'doc_id': list()}
            label_map = self.process_data_obj.labels if not self.is_gold else list(self.dataset_obj.experimental_label_map.values())

            doc_stats.update({label: list() for label in label_map})
            doc_stats.update({'overall': list(), 'result_type': list()})


            for doc_name, doc_id in self.process_data_obj.map.items():
                doc_data = self.process_data_obj.doc_data[doc_id]

                doc_stats['doc_id'].append(doc_name)
                label_counter = {label:0 for label in label_map}
                lab_count_auto = Counter([each['label'] for each in doc_data['argument_labels']])
                auto_counter = lab_count_auto if not self.is_gold else self.get_experimental_arguments(lab_count_auto)
                label_counter.update(auto_counter)
                doc_stats['result_type'].append(self.get_result_type(doc_data['argument_labels']))

                for lab, val in label_counter.items():
                    doc_stats[lab].append(val)

                formalistic_info = 'formalistic' if '- FORMALISTIC' in doc_data['f_labels'][0]['label_ui'] else 'non-formalistic'
                doc_stats['overall'].append(formalistic_info)

            df = pd.DataFrame(doc_stats)
            df.to_csv(arg_stats_file, index=False)

        df = pd.read_csv(arg_stats_file, index_col=False)
        return df




    def compute_kappa(self, arg_p_doc: pd.DataFrame) -> float:
        """
        Method is used to compute inter-annotator agreement score, kappa, using following steps:
        1. Walk through all documents in the given documents;
        2. For each document do:
            2.1. Collect annotations that were extracted from the INCePTION;
            2.2. Extract annotation per each annotator -> IF there is None case (at least 1 annotator didn't annotate),
                then discard the document from the computation process;
            2.3. Combine all of these annotations in one dictionary (i.e., the dictionary that holds all documents);
        3. Create a confusion table based on annotators' annotations;
        4. Compute the observed value by using the confusion table;
        5. Compute the expected value by using the confusion table;
        6. Compute the kappa using the observed and expected values, were computed at the 4th and 5th steps,
            respectively.

        :param arg_p_doc: DataFrame object that includes formalistic label information for all documents
        :return: kappa score that is computed based on aforementioned steps
        """

        label_dict = {'formalistic': 1, 'non_formalistic': 0}

        annotation_result = {annotator: list() for annotator in self.reader_obj.annotators}

        for doc_name in arg_p_doc['doc_id']:
            doc_idx = self.process_data_obj.map[doc_name]
            annotations = self.process_data_obj.doc_data[doc_idx]['annotations']
            current_val = dict()
            for annotator, annotation in annotations.items():
                if annotation['formalistic_labels']:
                    current_val[annotator] = label_dict['non_formalistic' if 'NON ' in annotation['formalistic_labels'][0]['label_ui'] else 'formalistic']
                else:
                    current_val[annotator] = None

            if None in current_val.values():
                continue

            for annotator, f_label in current_val.items():
                annotation_result[annotator].append(f_label)
        confusion_array = self.confusion_table(annotation_result, label_dict)
        observed = self.compute_observed(confusion_array)
        expected = self.compute_expected(confusion_array)
        kappa = (observed - expected) / (1 - expected)
        sns_map = sns.heatmap(confusion_array, annot=True, xticklabels=['NF', 'F'], yticklabels=['NF', 'F'])
        sns_map.get_figure().savefig(os.path.join(self.stats_folder, 'annotator_heatmap.png'))
        plt.close()
        return kappa

    def collect_disagreements(self, arg_p_doc: pd.DataFrame):
        pass

    @staticmethod
    def compute_observed(confusion_array: np.array) -> float:
        """
        Method is used to compute observed value, which is the trace of the matrix divided by sum of all elements;
        :param confusion_array: Confusion table that represents annotators' annotations with respect to each other;
        :return: a float number for the observed value
        """
        return confusion_array.trace() / confusion_array.sum()

    @staticmethod
    def compute_expected(confusion_array: np.array) -> float:
        """
        Method is used to compute the expected value using the confusion array
        :param confusion_array: Confusion table that represents annotators' annotations with respect to each other;
        :return: a float number for the expected value
        """
        p_e = 0
        for idx in range(confusion_array.shape[0]):
            row_prob = confusion_array[idx, :].sum() / confusion_array.sum()
            col_prob = confusion_array[:, idx].sum() / confusion_array.sum()
            p_e += row_prob * col_prob

        return p_e

    def confusion_table(self, annot_result: dict, labels: dict) -> np.array:
        """
        Method is used to generate a confusion table, which contains annotators' annotations with respect to each other
        :param annot_result: Dictionary that contains annotations per annotator
        :param labels: dictionary that holds labels that annotators annotated
        :return: a numpy array for the confusion table
        """
        annot_1 = self.reader_obj.annotators[0]
        annot_2 = self.reader_obj.annotators[1]

        confusion_array = np.zeros(shape=(len(labels), len(labels)))
        for v, v_2 in zip(annot_result[annot_1], annot_result[annot_2]):
            confusion_array[v, v_2] += 1

        return confusion_array


    @staticmethod
    def get_arguments(data: dict) -> dict:
        """
        Method is used to simplify formalistic labels
        :param data: annotation dictionary, where the keys are annotators, and values are their annotation list
        :return: a new dictionary for simplified annotations
        """
        formalistic_results = dict()
        for annotator, annotations in data.items():
            formalistic_label = 'formalistic' \
                if 'NON - FORMALISTIC' in annotations['formalistic_labels'][0]['label_ui'] else 'non-formalistic'
            formalistic_results[annotator] = formalistic_label

        return formalistic_results

    def num_args_hist(self, arguments_per_doc):

        non_arguments = ['RRESULT', 'overall', 'result_type', 'doc_id']
        args_dict = arguments_per_doc.copy()

        arguments = [each for each in arguments_per_doc.columns if each not in non_arguments]
        args_dict['num_args'] = args_dict.loc[:, arguments].sum(axis=1)
        sns_hist = sns.histplot(args_dict[['doc_id', 'num_args']], x='num_args', bins=30)
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_xlabel('Number of tokens', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        fig.set_size_inches(w=7, h=5)
        num_args = list(args_dict['num_args'])
        sns_hist.set_title(f'Number of arguments over all documents (μ = {int(sum(num_args) / len(num_args))})', fontdict={'fontsize': 18})
        sns_hist.set(xlabel='Number of arguments')
        sns_hist.get_figure().savefig(os.path.join(self.stats_folder, 'num_args_hist.pdf'))
        plt.close()

        return args_dict

    def num_tokens_hist(self):
        num_tokens_dict = {'doc_id': [], 'num_tokens': []}
        for doc_name, doc_id in self.process_data_obj.map.items():
            doc_data = self.process_data_obj.doc_data[doc_id]
            num_tokens = len(doc_data['tokens'])

            num_tokens_dict['doc_id'].append(doc_name)
            num_tokens_dict['num_tokens'].append(num_tokens)

        num_toks = num_tokens_dict['num_tokens']
        stats_frame = pd.DataFrame(num_tokens_dict)
        sns_hist = sns.histplot(stats_frame, x='num_tokens', bins=50)
        sns.set(rc = {'figure.figsize': (3, 1)})
        ax = plt.gca()
        fig = plt.gcf()
        fig.set_size_inches(w=7, h=5)
        sns_hist.set_title(f'Number of tokens over all documents (μ = {int(sum(num_toks) / len(num_toks))})', fontdict={'fontsize': 18})
        # sns_hist.set(xlabel='Number of tokens')
        ax.set_xlabel('Number of tokens', fontsize=18)
        ax.set_ylabel('Count', fontsize=18)
        sns_hist.get_figure().savefig(os.path.join(self.stats_folder, 'num_tokens_hist.pdf'))
        plt.close()
        print('Tokens in documents:')
        print(f'minimum number of tokens: {min(num_toks)}')
        print(f'maximum number of tokens: {max(num_toks)}')
        print(f'mean number of tokens: {np.mean(num_toks)}')

        # annotations = self.process_data_obj.doc_data[doc_idx]['annotations']

    def argument_statistics(self):
        labels = {k: {'doc_id': list(), 'freq': list()} for k in sorted(set(self.dataset_obj.experimental_label_map.values())) if k!='RRESULT'}
        unique_labels = {k: {'doc_id': list(), 'freq': list()} for k in sorted(set(self.dataset_obj.experimental_label_map.values())) if k!='RRESULT'}

        overall = {label: list() for label in labels.keys()}
        overall.update({'doc_id': list(), 'pars_no_args': list(), 'pars_one_arg': list(), 'pars_more_args': list(), 'num_pars': list(), 'num_args': list()})
        for split, dataset in self.dataset_obj.dataset.items():
            for idx, sample in enumerate(dataset['doc_id']):
                doc_labels = list()
                pars_no_args = 0
                pars_one_arg = 0
                pars_more_args = 0
                for paragraph in dataset['labels'][idx]:

                    if not paragraph:
                        pars_no_args += 1
                    else:
                        if len(paragraph) == 1:
                            pars_one_arg += 1
                        else:
                            pars_more_args += 1

                    doc_labels.extend(paragraph)

                counts = Counter(doc_labels)
                overall['doc_id'].append(sample)

                # unique_per_doc = list(set(doc_labels))
                num_args = sum(counts.values())
                # input(f'{counts.values()}, num_args={num_args}')
                for argument in labels.keys():
                    overall[argument].append(counts[argument])
                    labels[argument]['freq'].append(counts[argument] if argument in counts.keys() else 0)
                    labels[argument]['doc_id'].append(sample)
                    unique_labels[argument]['doc_id'].append(sample)
                    unique_labels[argument]['freq'].append(1) if argument in counts.keys() else 0
                overall['pars_no_args'].append(pars_no_args)
                overall['pars_one_arg'].append(pars_one_arg)
                overall['pars_more_args'].append(pars_more_args)
                overall['num_pars'].append(len(dataset['labels'][idx]))
                overall['num_args'].append(num_args)

        df = pd.DataFrame(overall)
        df.to_csv(os.path.join(self.stats_folder, 'par_doc_stats_latest.csv'), index=False)

        print(f"Number of arguments overall: {sum(overall['num_args'])}")
        print(f"Documents with no arguments: {Counter(overall['num_args'])[0]}")
        print(f"Number of paragraphs with no arguments: {sum(overall['pars_no_args'])}")

        print(f'minimum number of arguments: {df["num_args"].min()}')
        print(f'maximum number of arguments: {df["num_args"].max()}')
        print(f'average number of arguments: {df["num_args"].mean()}')

        print(f"Number of paragraphs overall: {sum(overall['num_pars'])}")
        print(f"Number of paragraphs with no arguments: {sum(overall['pars_no_args'])}")
        print(f"Number of paragraphs with one arguments: {sum(overall['pars_one_arg'])}")
        print(f"Number of paragraphs with at least 2 arguments: {sum(overall['pars_more_args'])}")

        return df

    def binary_coexistence(self):
        labels_set = [k for k in sorted(set(self.dataset_obj.experimental_label_map.values())) if k != 'RRESULT']
        labels_idx = {k: idx for idx, k in enumerate(labels_set)}
        co_existence = np.zeros((len(labels_idx.keys()), len(labels_idx.keys())))

        for split, dataset in self.dataset_obj.dataset.items():
            for idx, sample in enumerate(dataset['doc_id']):
                for paragraph in dataset['labels'][idx]:
                    if len(paragraph) == 1:

                        row_idx = labels_idx[paragraph[0]]
                        co_existence[row_idx, row_idx] += 1


                    elif len(paragraph) > 1:

                        for par_row in paragraph:
                            for par_col in paragraph:
                                if par_row != par_col:
                                    row_idx = labels_idx[par_row]
                                    col_idx = labels_idx[par_col]
                                    co_existence[row_idx, col_idx] += 1

                    else:
                        continue



        print(co_existence)

        sns.heatmap(co_existence, annot=True, fmt='g', cmap='Blues', xticklabels=list(labels_idx.keys()),
                    yticklabels=list(labels_idx.keys()))
        plt.title('Binary co-existence of Argument Labels')
        plt.xlabel('Label')
        plt.ylabel('Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.stats_folder, 'coexistence_heatmap.pdf'))
        plt.show()
        plt.close()

    def process(self) -> None:
        """
        Method is used as a __main__ of the class, which combines all relevant information
        :return: None is returned
        """
        arg_per_doc = self.get_args_per_doc()
        kappa = self.compute_kappa(arg_per_doc)
        self.binary_coexistence()
        self.num_args_hist(arg_per_doc)
        self.num_tokens_hist()
        self.argument_statistics()


        # print('kappa:', kappa)
