import glob
import logging
import os
import sys
import threading


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
from market_data.contract_handler.contract_handler import ContractHandler

from quantlib.calculation.analytics.measure import RiskMeasure
from quantlib.calculation.analytics.position.instruments.interest_rate.european import VanillaImpliedVolatility, \
    VanillaEuropean
from quantlib.calculation.analytics.position.pricer.pricer import BlackScholesValuationModelSingleAssetAnalytic
from quantlib.calendar.holidays import HolidayCalendar, gbp_calendar

warnings.filterwarnings('ignore')


def data_to_distribution_calc(input_file: str, output_path: str, calendar: HolidayCalendar, model_date: date):
    output_chunk_suc_df_list = []
    output_chunk_fail_df_list = []
    file_name = input_file.split('/')[-1]

    option_data_output_path = f"./output/option_data{output_path.split('output')[-1]}"
    distribution_output_path = f"./output/distribution{output_path.split('output')[-1]}"
    failed_data_output_path = f"./output/fail_data{output_path.split('output')[-1]}"

    try:
        os.makedirs(option_data_output_path, exist_ok=True)
        os.makedirs(distribution_output_path, exist_ok=True)
        os.makedirs(failed_data_output_path, exist_ok=True)
    except Exception as e:
        print(f'Failed to create directory: {e}')
        pass

    option_data_output_file = os.path.join(option_data_output_path, file_name.split('.csv')[0] + '_option.csv')
    distribution_suc_output_file = os.path.join(distribution_output_path,
                                                file_name.split('.csv')[0] + '_distribution.csv')
    distribution_fail_output_file = os.path.join(failed_data_output_path, file_name.split('.csv')[0] + '_fail.csv')

    def _process_chunk_data(row_data: pd.DataFrame, no_chunk: int) -> None:

        current_chunk_suc_result_df = pd.DataFrame()
        current_chunk_fail_result_df = pd.DataFrame()

        n_s = 0
        n_f = 0

        fail_contract_dict = {}
        total_num = row_data.shape[0]
        for _, row in row_data.iterrows():
            contract_dict = row.to_dict()
            try:
                row_do = rowdata_dto(row, DataTypes.QUOTE)
                contract = row_do.to_contract()
                if contract.contract_type == ContractType.Option:
                    maturity = contract.get_contract_maturity_dates_by_contract_id(
                        model_date, [calendar], terminal_rule
                    )
                    contract_dict.update(contract.to_dict())
                    contract_dict['maturity'] = maturity

                    vol_instrument = VanillaImpliedVolatility(
                        price=row_do.get_price(),
                        strike=contract.get_strike() / 100,
                        expiration_date=maturity,
                        option_type=contract.get_option_type(),
                        delivery_date=maturity,
                        underlying=contract.get_underlying()
                    )
                    spot = 1
                    vol = 0.3
                    r = q = 0.1
                    valuation_datetime = row_do.get_data_time()

                    pricer = BlackScholesValuationModelSingleAssetAnalytic(
                        instrument=vol_instrument,
                        spot=spot,
                        vol=vol,
                        r=r,
                        q=q,
                        valuation_date=valuation_datetime.date(),
                        valuation_time=valuation_datetime.time()
                    )

                    implied_vol = pricer.pv()

                    vanilla_instrument = VanillaEuropean(
                        strike=contract.get_strike() / 100,
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
                        valuation_date=valuation_datetime.date(),
                        valuation_time=valuation_datetime.time()
                    )
                    density = pricer.calc(RiskMeasure.DENSITY)

                    contract_dict['Vol'] = implied_vol
                    contract_dict['Density'] = density
                    contract_dict['Valuation_Time'] = valuation_datetime.isoformat()

                    current_chunk_suc_result_df = current_chunk_suc_result_df.append(contract_dict, ignore_index=True)
                    n_s += 1
                else:
                    fail_contract_dict['#RIC'] = contract_dict['#RIC']
                    fail_contract_dict['Error'] = 'Not an option contract'
                    current_chunk_fail_result_df = current_chunk_fail_result_df.append(fail_contract_dict,
                                                                                       ignore_index=True)
                    n_f += 1
                    continue
            except Exception as e:
                fail_contract_dict['#RIC'] = contract_dict['#RIC']
                fail_contract_dict['Error'] = str(e)
                current_chunk_fail_result_df = current_chunk_fail_result_df.append(fail_contract_dict,
                                                                                   ignore_index=True)
                print(f'Failed to process option contract [{n_f}] in file [{input_file}], error: {str(e)}')
                n_f += 1
                continue
        output_chunk_suc_df_list.append(current_chunk_suc_result_df)
        output_chunk_fail_df_list.append(current_chunk_fail_result_df)
        print(f'Finished processing chunk [{no_chunk}], total: [{total_num}], success: [{n_s}], fail: [{n_f}]')

    terminal_rule = ContractTerminationRule.ThirdWednessday

    raw_data = pd.read_csv(input_file)
    raw_data['id_len'] = raw_data['#RIC'].apply(lambda x: len(x))
    option_data = raw_data[raw_data['id_len'] > 6]
    del raw_data

    option_data.to_csv(option_data_output_file, index=False)
    option_data = ContractHandler.quote_data_handler(option_data)
    option_data = option_data.drop(['id_len'], axis=1)

    # raw_data.shape[0]
    total = option_data.shape[0]
    thread_name = threading.current_thread().name

    logging.info(f'Loaded file, start to process file [{input_file}]')
    print(f'Loaded file, start to process file [{input_file}], Thread: [{thread_name}]')

    # num_of_row_in_chunk = 2048

    # num_of_chunk = total // 2048 + 1

    num_of_chunk = 128
    num_of_row_in_chunk = total // 128 + 1

    threads = []
    print(f'Allocating threads: [{num_of_chunk}]')
    for i in range(num_of_chunk):
        start = i * num_of_row_in_chunk
        end = (i + 1) * num_of_row_in_chunk
        if end > total:
            end = total
        chunk_data = option_data.iloc[start: end]

        threads.append(threading.Thread(target=_process_chunk_data, args=(chunk_data, i + 1)))

    print('Start processing threads')
    for thread in threads:
        try:
            thread.start()
        except Exception as e:
            print(f'Failed to start thread: [{str(e)}]')
            continue

    print('Waiting for threads to finish')
    for thread in threads:
        thread.join()

    print('All threads finished')

    output_suc_df = pd.DataFrame()
    for df in output_chunk_suc_df_list:
        output_suc_df = pd.concat([output_suc_df, df], axis=0)

    output_suc_df.to_csv(distribution_suc_output_file, index=False)

    output_fail_df = pd.DataFrame()
    for df in output_chunk_fail_df_list:
        output_fail_df = pd.concat([output_fail_df, df], axis=0)
    output_fail_df.to_csv(distribution_fail_output_file, index=False)

    print(f'Finished processing file: [{input_file}], output file: [{output_path}]')


if __name__ == '__main__':
    cld = gbp_calendar()
    current_input_path = './input/'
    # years = ['2020', '2021', '2022', '2023']
    years = ['2020']
    contract_name = '0#FLG+'

    input_paths = [os.path.join(current_input_path, contract_name, year) for year in years]

    current_output_path = './output'
    current_output_paths = [f'{current_output_path}/{contract_name}/{year}' for year in years]

    input_files = []
    output_paths = []

    for idx in range(len(input_paths)):
        input_files += glob.glob(os.path.join(input_paths[idx], "*.csv.gz"))
        output_paths += [current_output_paths[idx]] * len(input_files)

    model_date = date(2023, 12, 1)
    print(f'Allocated input files to processes')

    for idx in range(len(input_files)):
        print(f'Start to process file: [{input_files[idx]}]')
        data_to_distribution_calc(input_files[idx], output_paths[idx], cld, date(2023, 12, 1))
        print(f'Finished processing file: [{input_files[idx]}], output file: [{output_paths[idx]}')

    print('================================')
    print('Finished processing all files')

#  nohup python /data/python/server.py > python.log 2>&1 &




