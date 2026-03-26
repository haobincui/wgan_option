import logging
import os
import sys
import threading
import time
from datetime import date
from itertools import cycle
from typing import List

import torch

BASE_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, '/home/haobin_cui/option_data_processor')
# sys.path.insert(0, '/Users/haobincui/Documents/option_data_processor')

from scripts.data_to_distribution.data_to_distribution_algos.calc_distribution import calc_distribution
from scripts.data_to_distribution.data_to_distribution_algos.get_input_and_output_files import \
    get_input_and_output_files
from quantlib.calendar.daycount import DayCountBusN, bus_250_gbp

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


def allocate_files_to_threads(input_files: List[str],
                              output_files: List[str],
                              future_files: List[str],
                              num_threads: int
                              ) -> tuple:
    """
    Allocate input files to threads.

    :param input_files: target option data files list
    :param output_files: target output files list
    :param future_files: target future files list
    :param num_threads:
    :return: input_option__files_list, output_files_list, future_files_list
    """

    input_files_list = []
    output_files_list = []
    future_files_list = []

    num_files_per_thread = len(input_files) // num_threads if len(input_files) % num_threads == 0 \
        else len(input_files) // num_threads + 1

    for i in range(num_threads):
        start = i * num_files_per_thread
        end = (i + 1) * num_files_per_thread
        if start >= len(input_files):
            break
        input_files_list.append(input_files[start:end])
        output_files_list.append(output_files[start:end])
        future_files_list.append(future_files[start:end])

    return input_files_list, output_files_list, future_files_list


def run_files(input_file_paths: List[str],
              output_file_paths: List[str],
              future_files_paths: List[str],
              daycount: DayCountBusN,
              data_date: date,
              chunksize: int = 4_096_000 * 4,
              device: torch.device = None,
              compress: bool = False
              ):
    total = len(input_file_paths)
    no_s = 0
    no_f = 0

    for idx in range(total):
        try:
            logging.info(f'Processing the file: [{input_file_paths[idx]}] '
                         f'in [{threading.current_thread().getName()}]...')
            calc_distribution(input_file_paths[idx],
                              output_file_paths[idx],
                              future_files_paths[idx],
                              daycount,
                              data_date,
                              device=device,
                              compress=compress,
                              chunksize=chunksize)

            logging.info(f'Finished processing the file: [{input_file_paths[idx]}] '
                         f'in [{threading.current_thread().getName()}] ... ')
            no_s += 1

        except Exception as e:
            logging.warning(f'Error processing the file [{input_file_paths[idx]}] in '
                            f'[{threading.current_thread().getName()}]:'
                            f' [{e}]')
            no_f += 1
        logging.info(f'Success [{no_s}], failed [{no_f}] out of [{total}] ...')
    logging.info(f'Finished all {total} files in [{threading.current_thread().getName()}]...')
    return


if __name__ == '__main__':
    s = time.time()
    logging.info('Starting ...')
    contract_id = '0#TY+'
    data_date = date(2024, 3, 16)

    input_path = os.path.abspath('./input')
    output_path = os.path.abspath('./output')
    years = ['2023']

    input_files, output_files, future_files = get_input_and_output_files(input_path,
                                                                         output_path,
                                                                         contract_id,
                                                                         years,
                                                                         get_future_file=True)

    print(f'Total input files: {len(input_files)}')
    daycount = bus_250_gbp

    if torch.cuda.is_available():
        devices = cycle([torch.device("cuda:0"), torch.device("cuda:1")])
    else:
        devices = cycle([torch.device("cpu")])

    num_threads = 4

    input_files_list, output_files_list, future_files_list = allocate_files_to_threads(input_files,
                                                                                       output_files,
                                                                                       future_files,
                                                                                       num_threads)

    threads = []
    compress = True
    chunk_size = 4_096_000 * 4

    print('Allocating input files to threads ...')
    for idx in range(len(input_files_list)):
        current_device = next(devices)
        t = threading.Thread(target=run_files, args=(
            input_files_list[idx], output_files_list[idx], future_files_list[idx],
            daycount, data_date, chunk_size, current_device, compress))
        t.setName(f'Thread-[{idx}]_[{current_device.type}:{current_device.index}]')

        threads.append(t)
    # logging.info('Start processing threads...')
    print('Start processing ...')
    for thread in threads:
        thread.start()
    print('Waiting for all threads to finish ...')
    for thread in threads:
        thread.join()
    e = time.time()
    logging.info(f'Finished all in [{e - s}] seconds! ! ')
    # print(f'[{e}]: Finished all in [{e - s}] seconds...')
