import glob
import pandas as pd
import os
import sys


BASE_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, '/home/haobin_cui/option_data_processor')

from market_data.contract_handler.contract_handler import ContractHandler
from connection.utils.file_processor.file_processor import FileProcesser


def get_option_data_from_raw_data(input_file: str,
                                  output_file: str,
                                  future_data_output_file: str = None,
                                  chunk_size: int = 4_096_000 * 4,
                                  compress: bool = True
                                  ):
    first_chunk = True
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        chunk['contract_type'] = chunk['#RIC'].apply(lambda x: ContractHandler(x).get_contract_type())
        option_data = chunk[chunk['contract_type'] == 'Option']

        option_data = option_data.drop(columns=['Alias Underlying RIC', 'Domain'])
        if future_data_output_file:
            future_data = chunk[chunk['contract_type'] == 'Future']
            future_data = future_data.drop(columns=['Alias Underlying RIC', 'Domain'])
        else:
            future_data = None
        del chunk

        if first_chunk:
            option_data.to_csv(output_file, index=False, mode='w')
            if future_data is not None:
                future_data.to_csv(future_data_output_file, index=False, mode='w')
            first_chunk = False
        else:
            option_data.to_csv(output_file, index=False, mode='a', header=False)
            if future_data is not None:
                future_data.to_csv(future_data_output_file, index=False, mode='a', header=False)
    if compress:
        print(f'Compressing file: [{output_file}] ... ')
        FileProcesser.compress(output_file)

        # delete the original file
        os.remove(output_file)
        if future_data_output_file is not None:
            FileProcesser.compress(future_data_output_file)
            os.remove(future_data_output_file)
        print(f'Compressed file: [{output_file}]')
        return
    else:
        return


if __name__ == '__main__':
    contract_id = '0#TY+'
    # contract_id = '0#FLG'
    years = ['2021']
    output_files = []
    future_output_files = []
    input_files = []
    for year in years:
        input_file_list = glob.glob(f'./input/raw_data/{contract_id}/{year}/*.csv.gz')
        print(f'Found [{len(input_file_list)}] raw data files, start processing ...')
        os.makedirs(f'./output/option_data/{contract_id}/{year}', exist_ok=True)
        os.makedirs(f'./output/future_data/{contract_id}/{year}', exist_ok=True)
        for input_file in input_file_list:
            input_files.append(input_file)
            output_file = input_file.split('/')[-1].split('.csv.gz')[0] + '_option.csv'
            output_files.append(f'./output/option_data/{contract_id}/{year}/{output_file}')
            future_file = input_file.split('/')[-1].split('.csv.gz')[0] + '_future.csv'
            future_output_files.append(f'./output/future_data/{contract_id}/{year}/{future_file}')

    for input_file, output_file, future_output_file in zip(input_files, output_files, future_output_files):
        print(f'Processing [{input_file}] to [{output_file}] ...')
        get_option_data_from_raw_data(input_file, output_file, future_output_file, compress=True)
        print(f'Finish [{input_file}] to [{output_file}] ...')
    print('Finished all !!!')




