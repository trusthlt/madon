import os
import pickle
import copy
import pandas as pd

from ast import literal_eval
from collections import Counter



class ProcessFeatures:
    def __init__(self, dataset_obj, parameters):
        self.parameters = parameters
        self.check_details()
        self.dataset_obj = dataset_obj
        self.descriptive_set = self.get_descriptive_set()
        self.features, self.lab2id = self.__main__()


    def check_details(self):
        if not self.parameters["split"]:
            raise NotImplementedError('Feature Engineering part is done only with the split dataset (fine-grained')
        if not self.parameters['is_gold']:
            raise NotImplementedError('Feature Engineering part is done only with the experimental labels')

    def get_descriptive_set(self):
        descriptive_dataset_path = os.path.join(self.dataset_obj.result_path, 'descriptive-'+self.dataset_obj.ds_name + '.csv')
        descriptive_dataset = pd.read_csv(
            descriptive_dataset_path,
            converters={
                'labels': literal_eval, 'result_labels': literal_eval, 'argument_info': literal_eval, 'tokenized_text': literal_eval
            } # it loads as string, this part does not allow it
        )
        return descriptive_dataset

    def process_features(self):
        num_labels = len(set(self.dataset_obj.experimental_label_map.values())) - 1
        feature_datasets = dict()
        for ds_type in ['train', 'dev', 'test']:
            dataset = self.dataset_obj.dataset[ds_type]
            feature_datasets[ds_type] = self.process_split_data(dataset, num_labels)
        return feature_datasets

    def process_split_data(self, dataset, num_labels):
        current_dataset = dict()

        dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')] # eliminate the column of Unnamed
        for idx in dataset['doc_id']:
            element = dataset[dataset['doc_id'] == idx]
            features, label_counts = self.process_single_element(idx, element)

            if not current_dataset: # if still empty
                current_dataset = {feature: [value] for feature, value in features.items()}
                continue
            for feature, value in features.items():
                current_dataset[feature].append(value)

        return current_dataset

    def get_raw_data_features(self, document_id):

        # because we produce doc_id with document names but keep the doc_id, doc_name structure only in descriptive set,
        # here we match data with document_name. In the gold standard datasets, doc_id is the doc_name in descriptive sets.
        subset = self.descriptive_set[self.descriptive_set['doc_name'] == document_id]
        arguments = subset['argument_info'].item() # to get specific data from the frame row
        arguments_token_size = 0
        num_arguments = 0

        for each in arguments:
            arguments_token_size += sum([len(argument['tokens']) for argument in each])
            num_arguments += len(each)

        avg_arg_len = arguments_token_size / num_arguments if num_arguments else 0 # how many tokens are in the argument
        num_tokens = sum([len(paragraph) for paragraph in subset['tokenized_text'].item()]) # number of tokens in the given text (all)
        return  num_tokens, avg_arg_len

    def process_single_element(self, document_id, instance):
        num_tokens_in_doc, avg_arg_len_in_doc = self.get_raw_data_features(document_id)
        result = {
            'doc_id': document_id, # document_id
            # 'doc_name': doc_name,
            'holistic_target': instance['f_labels'].item(),
            'doc_len_tokens': num_tokens_in_doc, # number of tokens in the document
            'num_args': sum(len(par) for par in instance['labels'].item()), # number of arguments in the document
            'avg_arg_len': avg_arg_len_in_doc, # average argument len in the document
        }

        all_labels = list()
        for label in instance['labels'].item():
            all_labels.extend(label)
        label_count = Counter(all_labels)
        all_labels_count = sum(label_count.values())
        label_list = list(sorted(self.dataset_obj.experimental_label_map.values())) #by sorting making sure of the order
        for each_label in label_list:
            if each_label == 'RRESULT':
                continue
            result[f'frac_arg_{each_label}'] = label_count[each_label] / all_labels_count if each_label in label_count.keys() else 0

        return result, label_count

    def __main__(self):
        features_path = os.path.join(self.dataset_obj.result_path, 'features.pickle') # since we do not run this with gold
        if not os.path.exists(features_path):
            dataset = self.process_features()
            # labels2idx = {label: idx for idx, label in enumerate(set(dataset['train']['holistic_target']))}
            features_data = {
                'dataset': dataset,
                'lab2id': {'O - OVERALL - FORMALISTIC': 1, 'O - OVERALL - NON FORMALISTIC': 0}
            }
            with open(features_path, 'wb') as features:
                pickle.dump(features_data, features)
        with open(features_path, 'rb') as features:
            features_data = pickle.load(features)

        return features_data['dataset'], features_data['lab2id']


    def get_dataset(self, ds_type, tabular: bool=False, inference: bool=False, seed_info: int= 42):
        ignore_list = ['doc_id', 'holistic_target']

        dataset = pd.DataFrame(self.features[ds_type]) if not inference else self.inference_features(ds_type, seed_info)


        data = dataset.drop(labels=ignore_list, axis=1)
        labels = dataset['holistic_target'].transform(lambda x: self.lab2id[x])
        if tabular:
            return data, labels

        return data.to_numpy(), labels.to_numpy()

    def inference_features(self, ds_type, seed_info):

        # 2nd scenario
        # inference_path = os.path.join(self.parameters['data_path'], 'gold_data', 'inference_data',
        #                        'paragraph-multilabel-llama-base-full-finetune-asy-seed-32_predicted_label_counts.pkl')

        # 1st scenario
        # inference_path = os.path.join(self.parameters['data_path'], 'gold_data', 'inference_data',
        #                               'paragraph-multilabel-llama-base-full-finetune-asy-seed-52_filtered_predicted_label_counts.pkl')

        # 3rd scenario
        filename = self.getfname(seed_info)

        inference_path = os.path.join(self.dataset_obj.ds_path, 'gold', 'inference_data',filename)
        with open(inference_path, 'rb') as f:
            data = pickle.load(f)
        if ds_type != 'test':
            raise NotImplementedError('Inference is done only with test dataset')

        feature_set = copy.deepcopy(self.features[ds_type])

        for idx, doc_id in enumerate(feature_set['doc_id']):
            overall = sum(data[doc_id].values())
            subframe = self.descriptive_set[self.descriptive_set['doc_name'] == doc_id]
            # print('bach')
            # print(subframe)
            tok_text  = subframe['tokenized_text'].item()
            args_infos = subframe['argument_info'].item()
            all_pars_doc = 0
            arg_par_doc = 0
            for paragraph, arginfo in zip(tok_text, args_infos):
                par_len = sum([len(sentence) for sentence in paragraph])
                all_pars_doc += par_len
                arg_par_doc += sum([len(paragraph[idx]) for idx, argss in enumerate(arginfo) if argss])


            num_arguments = 0
            for argument, count in data[doc_id].items():
                feature_set[f'frac_arg_{argument}'][idx] = count / overall if overall else 0
                num_arguments += count

            feature_set['num_args'][idx] = num_arguments

            avg_par_len = all_pars_doc / len(tok_text)
            feature_set['avg_arg_len'][idx] = avg_par_len if feature_set['num_args'][idx] else 0

        return pd.DataFrame(feature_set)

    def getfname(self, seed_info):

        if self.parameters['inference_choice'] == 'finetune':
            return f'paragraph-multilabel-llama-base-full-finetune-asy-seed-{seed_info}_predicted_label_counts.pkl'
        elif self.parameters['inference_choice'] == 'finetune_filtered':
            return f'paragraph-multilabel-llama-base-full-finetune-asy-seed-{seed_info}_filtered_predicted_label_counts.pkl'
        else:
            raise NotImplementedError


