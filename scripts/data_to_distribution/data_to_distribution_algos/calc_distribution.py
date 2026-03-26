import logging
import os
import threading
# import time
from datetime import date

import pandas as pd
import torch

from connection.utils.file_processor.file_processor import FileProcesser
from quantlib.calendar.daycount import DayCountBusN
from scripts.data_to_distribution.data_to_distribution_algos.get_distribution import get_distribution
from scripts.data_to_distribution.data_to_distribution_algos.get_spot import get_spot_price_df


def calc_distribution(input_file_path: str, output_file_path: str, future_file_path: str,
                      daycount: DayCountBusN, data_date: date,
                      chunksize: int = 4_096_000 * 4, device: torch.device = None, compress: bool = False):

    start_date = date.fromisoformat(future_file_path.split('_')[-2])

    # get spot price map
    # s = time.time()
    # spot_price_map = get_spot_price_map(future_file_path, start_date)
    spot_price_df = get_spot_price_df(future_file_path, start_date)
    # e = time.time()
    # print(f'Loaded spot price map in [{e - s}] seconds ...')

    chunk_data = pd.read_csv(input_file_path, chunksize=chunksize)
    first_chunk = True
    print(f'Loaded data for [{input_file_path}] in [{threading.current_thread().getName()}]...')
    logging.info(f'Started processing the file [{input_file_path}]...')
    for chunk in chunk_data:
        if first_chunk:
            result = get_distribution(chunk, daycount, data_date, spot_price_df, device=device)
            result.to_csv(
                output_file_path,
                index=False,
                mode='w',
                encoding='utf-8'
            )
            first_chunk = False
        else:
            result = get_distribution(chunk, daycount, data_date, spot_price_df, device=device)
            result.to_csv(
                output_file_path,
                index=False,
                mode='a',
                header=False,
                encoding='utf-8'
            )
    logging.info(f'Finished processing the file [{input_file_path}] to [{output_file_path}]'
                 f'in [{threading.current_thread().getName()}] ...')
    if compress:
        FileProcesser.compress(output_file_path)
        print(f'Compressed the file: [{output_file_path}] in in [{threading.current_thread().getName()}]...')
        os.remove(output_file_path)
        print('Finished')
        return
    else:
        print(f'Finished')
        return
