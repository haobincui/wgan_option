
import os
import sys
from datetime import date

import torch

BASE_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, '/home/haobin_cui/option_data_processor')


from scripts.data_to_distribution.data_to_distribution_algos.calc_distribution import calc_distribution
from scripts.data_to_distribution.data_to_distribution_algos.get_input_and_output_files import \
    get_input_and_output_files
from connection.utils.file_processor.file_processor import FileProcesser
from quantlib.calendar.daycount import bus_250_gbp

if __name__ == '__main__':
    contract_id = '0#TY+'

    # contract_id = '0#FLG+'
    input_path = f'./output'
    output_path = './output'
    years = ['2023']

    input_files, output_files, future_files = get_input_and_output_files(input_path, output_path,
                                                                         contract_id, years,
                                                                         get_future_file=True)

    print(f'Total input files: {len(input_files)}')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    daycount = bus_250_gbp
    total = len(input_files)
    no_s = 0
    no_f = 0

    for input_file, output_file, future_file in zip(input_files, output_files, future_files):
        try:
            print(f'Processing the file: {input_file} ...')
            calc_distribution(input_file, output_file, future_files, daycount, date(2024, 3, 16),
                              device=device, compress=True)
            print(f'Finished processing the file: {input_file} ... ')
            no_s += 1
        except Exception as e:
            print(f'Error processing the file: {input_file} ...')
            print(f'Error: {e}')
            no_f += 1
        print(f'Success [{no_s}], failed [{no_f}] out of [{total}] ...')
    print('Finished all ...')
