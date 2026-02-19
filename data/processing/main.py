from utilities.utils import *
from read_data import ReadData
from process_raw_data import ProcessData
from create_dataset import CreateDataset
from data_statistics import DataStats
def __main__():
    parameters = get_parameters()

    # folder setup, will be used to read and collect data
    data_path = parameters['data_path']
    datasets_folder = os.path.join(data_path, 'raw_data')
    processed_dataset_folder = os.path.join(data_path, 'processed_datasets')
    data_statistics_folder = os.path.join(data_path, 'statistics')
    curated_path = os.path.join(datasets_folder, 'curated')
    raw_path = os.path.join(datasets_folder, 'annotated')

    annotators = ['matyas.bartak', 'vitek.eichler']


    # reader object to read and store raw data in one file for further uses
    reader_obj = ReadData(processed_dataset_folder, curated_path, raw_path, annotators) if parameters['read_data'] else None
    # processes the raw data for creating dataset for task-specific setup
    raw_data_process_obj = ProcessData(reader_obj, processed_dataset_folder)
    # dataset object to create either meta (with 15 arguments) or the gold (with 8 arguments) setup in MADON scenarios
    dataset_obj = CreateDataset(
        processed_dataset_folder, parameters['is_gold'], parameters['split']
    )

    if parameters['statistics']:
        # statistics object to perform some statistical analysis. It is optional, you need to set statistics parameter
        # true, in order to see the statistics process
        statistics_obj = DataStats(
            reader_obj, raw_data_process_obj, dataset_obj, parameters['is_gold'], data_statistics_folder
        )
        statistics_obj.process()

if __name__ == '__main__':
    __main__()
