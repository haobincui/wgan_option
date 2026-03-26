import glob
import logging
import os
import sys
import threading

from tqdm import tqdm

BASE_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, '/home/haobin_cui/option_data_processor')
# sys.path.insert(0, '/Users/haobincui/Documents/option_data_processor')

from datetime import date

import pandas as pd
import warnings

from market_data.contract_handler.contract_type import ContractType
from market_data.contract_handler.utils import ContractTerminationRule
from market_data.dto.contracts_dto import rowdata_dto, DataTypes
from quantlib.calculation.analytics.measure import RiskMeasure
from quantlib.calculation.analytics.position.instruments.interest_rate.european import VanillaImpliedVolatility, \
    VanillaEuropean
from quantlib.calculation.analytics.position.pricer.pricer import BlackScholesValuationModelSingleAssetAnalytic
from quantlib.calendar.holidays import HolidayCalendar, gbp_calendar

warnings.filterwarnings('ignore')


# def decompress_file(input_path: str, write: bool = False):
#     try:
#         FileProcesser.decompress(input_path, write=write)
#         logging.debug(f'Decompress file [{input_path}]')
#         return True
#
#     except Exception as e:
#         logging.debug(f'Failed to decompress file [{input_path}], error: {str(e)}')
#         return False
#

def data_to_distribution_calc(input_path: str, output_path: str, calendar: HolidayCalendar, model_date: date):
    # decompress_file(f, write=False)
    no_s_option = 0
    no_f_option = 0
    no_future = 0
    n = 0

    terminal_rule = ContractTerminationRule.ThirdWednessday

    output_df = pd.DataFrame()
    raw_data = pd.read_csv(input_path)
    raw_data['id_len'] = raw_data['#RIC'].apply(lambda x: len(x))
    option_data = raw_data[raw_data['id_len'] > 6]

    option_data_output_path = output_path.split('.csv')[0] + '_option.csv'
    option_data.to_csv(option_data_output_path, index=False)

    # raw_data.shape[0]
    total = option_data.shape[0]
    thread_name = threading.current_thread().name

    logging.info(f'Loaded file, start to process file [{input_path}]')
    print(f'Loaded file, start to process file [{input_path}], Thread: [{thread_name}]')
    for i in tqdm(range(0, total), desc=f'Thread: [{thread_name}]'):
        row = option_data.iloc[i]
        try:
            row_do = rowdata_dto(row, DataTypes.QUOTE)
            contract = row_do.to_contract()
            if contract.contract_type == ContractType.Option:
                maturity = contract.get_contract_maturity_dates_by_contract_id(
                    model_date, [calendar], terminal_rule
                )
                contract_dict = contract.to_dict()
                contract_dict['maturity'] = maturity

                vol_instrument = VanillaImpliedVolatility(
                    price=row_do.get_price(),
                    strike=contract.get_strike(),
                    expiration_date=maturity,
                    option_type=contract.get_option_type(),
                    delivery_date=maturity,
                    underlying=contract.get_underlying()
                )
                spot = 100
                vol = 0.3
                r = q = 0.1

                pricer = BlackScholesValuationModelSingleAssetAnalytic(
                    instrument=vol_instrument,
                    spot=spot,
                    vol=vol,
                    r=r,
                    q=q,
                    valuation_date=model_date
                )

                implied_vol = pricer.calc(RiskMeasure.PV)

                vanilla_instrument = VanillaEuropean(
                    strike=contract.get_strike(),
                    expiration_date=maturity,
                    option_type=contract.get_option_type(),
                    delivery_date=maturity,
                    underlying=contract.get_underlying()
                )
                pricer = BlackScholesValuationModelSingleAssetAnalytic(
                    instrument=vanilla_instrument,
                    spot=spot,
                    vol=implied_vol,
                    r=r,
                    q=q,
                    valuation_date=model_date
                )
                density = pricer.calc(RiskMeasure.DENSITY)

                contract_dict['Vol'] = implied_vol
                contract_dict['Density'] = density

                # output_df = pd.concat([output_df, contract_dict], axis=0)
                output_df = output_df.append(contract_dict, ignore_index=True)
                no_s_option += 1
                n += 1
                logging.debug(f'Success option contract :[{no_s_option}], [{total - n}] left')
                # print(f'Success option contract :[{no_s_option}], [{total - n}] left')
            else:
                no_future += 1
                n += 1
                logging.debug(f'Pass future contract :[{no_future}], [{total - n}] left')
                # pass
        except Exception as e:
            no_f_option += 1
            n += 1
            logging.critical(f'Failed to process option contract [{no_f_option}], error: {str(e)}')
            # print(f'Failed to process option contract [{no_f_option}], error: {str(e)}')
            continue

    distribution_output_path = output_path.split('.csv')[0] + '_distribution.csv'
    output_df.to_csv(distribution_output_path, index=False)

    print(f'Finished processing file: [{input_path}], output file: [{output_path}], '
          f'Thread: [{thread_name}]; '
          f'Option contracts: Success [{no_s_option}], Fail [{no_f_option}]; Future contracts: [{no_future}]]')


if __name__ == '__main__':

    cld = gbp_calendar()

    input_path = './input/'
    # years = ['2020', '2021', '2022', '2023']
    years = ['2023']
    contract_name = '0#FLG'

    input_paths = [os.path.join(input_path, year, contract_name) for year in years]

    output_path = './output'
    output_paths = [f'{output_path}/{year}/{contract_name}/' for year in years]

    input_files = []

    for idx in range(len(input_paths)):

        try:
            os.makedirs(output_paths[idx], exist_ok=True)
            # os.makedirs(input_paths[idx], exist_ok=True)
        except Exception as e:
            print(f'Failed to create directory: {e}')
            pass

        input_files += glob.glob(os.path.join(input_paths[idx], "*.csv.gz"))

    output_files = []
    for input_file in input_files:
        output_file = input_file.split('input')[0] + 'output' + input_file.split('input')[1]
        output_file = output_file.split('.gz')[0]
        output_files.append(output_file)

    model_date = date(2023, 12, 1)
    print(f'Allocated input files to processes')
    # run(input_files[idx], output_files[idx], cld, date(2023, 12, 1))

    total_files = len(input_files)
    num_file_groups = total_files // 4 + 1

    input_groups = []
    output_groups = []
    for i in range(num_file_groups):
        input_groups.append(input_files[i * 4: (i + 1) * 4])
        output_groups.append(output_files[i * 4: (i + 1) * 4])

    n_group = 0

    for input_group, output_group in zip(input_groups, output_groups):

        threads = [threading.Thread(target=data_to_distribution_calc, args=(input_file, output_file, cld, model_date))
                   for input_file, output_file in zip(input_group, output_group)]

        print(f'Start to process files')
        for thread in threads:
            thread.start()
        print(f'All threads started')

        for thread in threads:
            thread.join()
        print(f'All threads finished')
        n_group += 1
        print(f'finished group {n_group}')


    def error_callback(e):
        print(f'Process Error: {e}')


    # pool = Pool(4)
    # pool.map(run, input_files, output_files)
    # for input_file, output_file in zip(input_files, output_files):
    #     pool.apply_async(data_to_distribution_calc, args=(input_file, output_file, cld, model_date),
    #                      error_callback=error_callback)
    # pool.close()
    # pool.join()
    print('Finished processing all files')
