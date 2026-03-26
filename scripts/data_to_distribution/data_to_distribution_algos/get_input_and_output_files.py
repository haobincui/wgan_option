import glob
import os
from typing import List


def get_input_and_output_files(input_path: str, output_path: str, contract_id: str, years: List[str], get_future_file: bool = False) -> tuple:
    """
    Get the input and output files for the distribution calculation.

    :param input_path:
    :param contract_id:
    :param years:
    :param get_future_file:
    :return: input_files: /option_data/*.csv.gz, output_files: *_distribution.csv, future_files: /future_data/*.csv.gz
    """

    input_files = []
    output_files = []
    future_files = []

    for year in years:
        files = glob.glob(f'{input_path}/option_data/{contract_id}/{year}/*.csv.gz')
        for file in files:
            input_files.append(file)
            future_file = f'{input_path}/future_data/{file.split("option_data/")[-1].replace("+", ":")}'.replace('_option','')
            future_files.append(future_file)
            output_folder = f'{output_path}/distribution/{contract_id}/{year}'
            os.makedirs(output_folder, exist_ok=True)
            output_files.append(output_folder + '/' + file.split('/')[-1].split('.csv.gz')[0] + '_distribution.csv')
    if get_future_file:
        return input_files, output_files, future_files
    return input_files, output_files
