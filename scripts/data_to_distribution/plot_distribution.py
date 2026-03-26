import os
import sys


BASE_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, '/home/haobin_cui/option_data_processor')
from datetime import datetime
import glob
import logging

import numpy as np
import pandas as pd
from connection.utils.file_processor.file_processor import FileProcesser
from scipy.stats import skew, kurtosis

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


def process_maturity_group(chunk: pd.DataFrame):
    """
        headers:
    ContractId,ContractType,Maturity,
    AskSpot,AskPrice,AskSize,ASKStrike,AskQuoteTime,AskVol,AskDensity,
    BidSpot,BidPrice,BidSize,BidStrike,BidQuoteTIme,BidVol,BidDensity
    :param chunk:
    :return:
    """
    chunk['AskQuoteTime'] = pd.to_datetime(chunk['AskQuoteTime'], infer_datetime_format=True)
    chunk['BidQuoteTIme'] = pd.to_datetime(chunk['BidQuoteTIme'], infer_datetime_format=True)
    chunk['Maturity'] = pd.to_datetime(chunk['Maturity'], infer_datetime_format=True)
    ask_quote_year = chunk['AskQuoteTime'].iloc[0].year
    bid_quote_year = chunk['BidQuoteTIme'].iloc[0].year
    quote_year = ask_quote_year if not np.isnan(ask_quote_year) else bid_quote_year
    start_of_year = datetime(int(quote_year), 1, 1)

    # Calculate the 3-minute time intervals for each time column

    chunk['Ask_time_interval'] = (chunk['AskQuoteTime'] - start_of_year).dt.total_seconds() // 3
    chunk['AskTimeToMaturity'] = (chunk['Maturity'] - chunk['AskQuoteTime']).dt.total_seconds() / 60
    chunk['Bid_time_interval'] = (chunk['BidQuoteTIme'] - start_of_year).dt.total_seconds() // 3
    chunk['BidTimeToMaturity'] = (chunk['Maturity'] - chunk['BidQuoteTIme']).dt.total_seconds() / 60

    # Initialize a DataFrame to store aggregated results
    results = []

    # Aggregate for AskQuoteTime
    ask_grouped = chunk.groupby('Ask_time_interval').agg({
        'AskDensity': ['mean', 'sum', 'std', 'count', lambda x: skew(x, bias=False), lambda x: kurtosis(x, bias=False)],
        'AskVol': ['mean', 'sum', 'std'],
        'AskSize': ['mean', 'sum', 'std'],
        'AskTimeToMaturity': ['mean']
    })
    ask_grouped.columns = ['Ask_' + '_'.join(col).strip() for col in ask_grouped.columns.values]
    ask_grouped = ask_grouped.reset_index().rename(columns={'Ask_time_interval': 'time_interval'})
    ask_grouped['QuoteType'] = 'Ask'
    ask_grouped['AskDensity_total_count'] = ask_grouped['Ask_AskDensity_count'] * ask_grouped['Ask_AskSize_sum']

    results.append(ask_grouped)

    # Aggregate for BidQuoteTIme
    bid_grouped = chunk.groupby('Bid_time_interval').agg({
        'BidDensity': ['mean', 'sum', 'std', 'count', lambda x: skew(x, bias=False), lambda x: kurtosis(x, bias=False)],
        'BidVol': ['mean', 'sum', 'std', 'count'],
        'BidSize': ['mean', 'sum', 'std'],
        'BidTimeToMaturity': ['mean']
    })
    bid_grouped.columns = ['Bid_' + '_'.join(col).strip() for col in bid_grouped.columns.values]
    bid_grouped = bid_grouped.reset_index().rename(columns={'Bid_time_interval': 'time_interval'})
    bid_grouped['QuoteType'] = 'Bid'
    bid_grouped['BidDensity_total_count'] = bid_grouped['Bid_BidDensity_count'] * bid_grouped['Bid_BidSize_sum']

    results.append(bid_grouped)

    # Combine results from both groupings
    final_result = pd.concat(results, ignore_index=True)

    return final_result


def process_chunk(chunk):
    """Process each chunk by first grouping by 'Maturity'."""
    chunk['Maturity'] = pd.to_datetime(chunk['Maturity'], infer_datetime_format=True)

    # Process each maturity group separately
    processed_groups = []
    for _, group in chunk.groupby('Maturity'):
        processed_group = process_maturity_group(group)
        processed_groups.append(processed_group)

    # Combine all processed maturity groups into a single DataFrame
    final_result = pd.concat(processed_groups, ignore_index=True)
    return final_result


def plot_distribution(input_file: str, output_file: str, compress: bool = True, chunk_size: int = 4096000 * 4):
    """
    headers:
    ContractId,ContractType,Maturity,
    AskSpot,AskPrice,AskSize,ASKStrike,AskQuoteTime,AskVol,AskDensity,
    BidSpot,BidPrice,BidSize,BidStrike,BidQuoteTIme,BidVol,BidDensity
    :param chunk_size:
    :param input_file:
    :param output_file:
    :return:
    """
    # probability sensitment data: [https://probability.nl/modelling/psi/]
    logging.info("Starting data processing ...")

    first_chunk = True
    i = 0

    # Process the dataset in chunks
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        print(f"Processing chunk: [{i}] ...")
        processed_chunk = process_chunk(chunk)
        mode = 'w' if first_chunk else 'a'
        processed_chunk.to_csv(output_file, mode=mode, index=False, header=first_chunk)
        first_chunk = False
        i += 1
        print(f"Processed chunk: [{i}] ...")

    logging.info(f"Data processing complete and saved to [{output_file}], total chunks processed: [{i}] !")
    if compress:
        FileProcesser.compress(output_file)
        print(f'Compressed the file: [{output_file}] ...')
        os.remove(output_file)
        print('Finished')
        return
    else:
        print(f'Finished')
        return
        


if __name__ == '__main__':
    contract_id = '0#TY+'
    year = '2023'
    input_path = f"./output/distribution/{contract_id}/{year}/"
    input_files = glob.glob(input_path + "*_distribution.csv.gz")
    output_files = [file.replace(".csv.gz", "_grouped-distribution.csv") for file in input_files]
    print(f'[{len(input_files)}] files founded ...')
    for input_file, output_file in zip(input_files, output_files):
        logging.info(f'Processing [{input_file}] ...')
        plot_distribution(input_file, output_file)
        print(f"Processed: {input_file} and saved to: {output_file}")
        logging.info((f"Processed: {input_file} and saved to: {output_file}"))






