from datetime import date
from typing import List, Dict, Union, Tuple
from dataclasses import asdict, is_dataclass
import json

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from quantlib.calendar.daycount import DayCountBusN
from quantlib.calendar.holidays import HolidayCalendar
from .svi_surface import SviVolSurface
from .svi_algo import SviCalibrationQuasiExplicit, SviCalibration, SviCalibrationGatheral

import logging

logger = logging.getLogger(__name__)

tenor_bin = [1, 3, 5, 10, 21, 42, 63, 84, 110, 126, 147, 168, 189, 210, 231, 250]
tenor_boundary = np.array([0, 2, 4, 7, 15, 30, 52, 73, 97, 118, 136, 157, 178, 199, 220, 241])
tau_minimum = 0
tau_maximum = 255
_minimum_required_amount = 5


def _json_default(value):
    if isinstance(value, (date,)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Type {type(value)} is not JSON serializable")


def _to_json(data) -> str:
    return json.dumps(data, default=_json_default, ensure_ascii=False)


class ImpliedVolSurfaceAlgo:
    def __init__(self, valuation_date: date, spot: float, calendar: HolidayCalendar, days_in_year: int = 250):
        self.valuation_date = valuation_date
        self.spot = spot
        self.calendar = calendar
        self.days_in_year = days_in_year

    def convert_market_vol_data(self, raw_data: pd.DataFrame) -> Union[None, Dict[str, List[float]]]:
        raw_data = raw_data.sort_values('expire_date')
        r = raw_data.get('r', 0.0175)
        q = raw_data.get('q', 0)
        group_map = raw_data.groupby(by='expire_date').groups
        # global_median_vol = raw_data['implied_vol'].median()
        vols = []
        taus = []
        percent_strikes = []
        record_ids = []
        for day in group_map.keys():
            idxs = group_map.get(day)
            tau = self.calendar.count_business_days(
                self.valuation_date,
                day,
                include_start=False,
                include_end=True,
            )
            record_id = [raw_data['record_id'][idx] for idx in idxs]
            percent_strike = np.array(raw_data['strike1'][idxs]) / (
                    self.spot * np.exp((r - q) * tau / self.days_in_year))
            # vols with tau higher than 255 trading days and lower than 1 trading days are eliminated
            if tau <= tau_minimum:
                logger.debug(f'Eliminated record_ids [{record_id}] due to small tau: {tau}')
                continue
            if tau > tau_maximum:
                logger.debug(f'Eliminated record_ids [{record_id}] due to large tau: {tau}')
                continue
            vol_array = raw_data['implied_vol'][idxs].values
            # local_median_vol = np.median(vol_array)
            # adjusted tau and vols to selected tau intervals,
            if tau in tenor_bin:
                pass
            else:
                idx = np.searchsorted(tenor_boundary, tau, side='left')
                tau = tenor_bin[idx - 1]
            x = []
            # extreme vols adjustment using dbscan,choose the cluster of vols with the largest amounts
            for i in range(0, len(percent_strike)):
                x.append([percent_strike[i], vol_array[i]])

            db = DBSCAN(eps=0.05, min_samples=5).fit(X=x)
            labels = db.labels_
            unique_data = np.unique(labels, return_counts=True)
            cluster_idx = np.argmax(unique_data[1])
            cluster_number = unique_data[0][cluster_idx]
            labels_idx = np.logical_and(labels == cluster_number, True)
            eliminated_idx = np.logical_not(labels_idx)
            if np.any(eliminated_idx):
                n_noise = np.sum(eliminated_idx)
                eliminated_record_ids = np.array(record_id)[eliminated_idx].tolist()
                logger.debug(f'Estimated number of noise points: {n_noise}; '
                             f'Eliminated noise points with record ids: {eliminated_record_ids}')
            else:
                logger.debug('None noise point is eliminated ')

            vol_array = vol_array[labels_idx]
            percent_strike = percent_strike[labels_idx]

            atm_vols_idxs = np.logical_not(percent_strike != 1)
            if np.any(atm_vols_idxs):
                left_vol_idx = np.logical_not(atm_vols_idxs)
                left_vols = vol_array[left_vol_idx]
                atm_vols = vol_array[atm_vols_idxs]
                atm_vol = np.mean(atm_vols)
                vol_array = np.array(left_vols, atm_vol)
            vol = vol_array.tolist()

            record_ids.append(record_id)
            percent_strikes.append(percent_strike)
            vols.append(vol)
            taus.append(tau)

        # adjust for flat surface, return None if it is flat
        if np.all(np.logical_and(np.concatenate(vols) - np.mean(np.concatenate(vols)) == 0, True)):
            logger.warning(f'Flat vol surface with mean: {np.mean(np.concatenate(vols))}')
            return None

        # adjust for the number of each vols for different taus
        amount_of_vols = [len(vols[i]) for i in range(0, len(vols))]
        median_amount_of_vols = np.median(amount_of_vols)
        for idx, amount in enumerate(amount_of_vols):
            if amount < 0.2 * median_amount_of_vols or amount < _minimum_required_amount:
                if idx == 0:
                    logger.warning(f'Delete and refilled vols with term [{taus[idx]}] '
                                   f'due to lack of enough amounts of vols')
                    vols[0] = vols[1]
                    percent_strikes[0] = percent_strikes[1]
                elif idx == len(amount_of_vols):
                    logger.warning(f'Delete and refilled vols with term [{taus[idx]}] '
                                   f'due to lack of enough amounts of vols')
                    vols[-1] = vols[-2]
                    percent_strikes[-1] = percent_strikes[-2]
                else:
                    logger.warning(f'Delete and refilled vols with term [{taus[idx]}] '
                                   f'due to lack of enough amounts of vols')
                    vols[idx] = np.multiply(vols[idx + 1], np.sqrt(taus[idx] / self.days_in_year))
                    percent_strikes[idx] = percent_strikes[idx + 1]
        if len(vols) == 0:
            logger.warning('None vols left')
            return None
        result_dict = {'vols': vols,
                       'business_days': taus,
                       'percent_strikes': percent_strikes}
        return result_dict

    def get_calibrated_result(self, filtrated_data, use_qls_calibration: bool = True) -> SviCalibration:

        vols = filtrated_data['vols']
        percent_strike = filtrated_data['percent_strikes']
        business_days = filtrated_data['business_days']

        vol_daycount = DayCountBusN(
            f"BUS{int(self.days_in_year)}",
            self.calendar,
            self.days_in_year,
        )
        if use_qls_calibration:
            return SviCalibrationQuasiExplicit(vols=vols,
                                               percent_strikes=percent_strike,
                                               business_days=business_days,
                                               vol_daycount=vol_daycount,
                                               valuation_date=self.valuation_date)
        else:
            return SviCalibrationGatheral(vols=vols,
                                          percent_strikes=percent_strike,
                                          business_days=business_days,
                                          vol_daycount=vol_daycount,
                                          valuation_date=self.valuation_date)

    @staticmethod
    def save_svi_params(svi_params: Dict[str, List[float]], valuation_date: date, calendar: HolidayCalendar,
                        days_in_year: int = 250, underlying: str = '000905.SH', ) -> pd.DataFrame:

        vol_daycount = DayCountBusN(
            f"BUS{int(days_in_year)}",
            calendar,
            days_in_year,
        )
        svi_dataclass = SviVolSurface(valuation_date=valuation_date,
                                      svi_params=svi_params,
                                      vol_daycount=vol_daycount)

        svi_json = _to_json(svi_dataclass)
        df = pd.DataFrame()
        df['instrument_id'] = [underlying]
        df['valuation_date'] = [valuation_date]
        df['svi_params'] = [svi_json]
        # df.to_csv('./svi_params.csv', columns=['instrument_id', 'valuation_date', 'svi_params'], index=False)
        return df

    @staticmethod
    def save_implied_vol_surface(vols: List[List[float]], percent_strikes: List[float], spot: float,
                                 times_to_maturity: List[int], days_in_year: int, valuation_date: date,
                                 underlying: str = '000905.SH') -> pd.DataFrame:
        values = []
        for idx, t in enumerate(times_to_maturity):
            vols_information = []
            for i, vol in enumerate(vols[idx]):
                vol_information = {'quote': vol,
                                   'percent': percent_strikes[i],
                                   'strike': None,
                                   'label': f'{int(percent_strikes[i] * 100)}% SPOT'}
                vols_information.append(vol_information)
            value = {'tenor': f'{int(t)}D',
                     'vols': vols_information,
                     'expiry': None}
            values.append(value)

        output = {'source': 'OFFICIAL',
                  'fittingModels': '[]',
                  'modelInfo': {
                      'modelName': 'TRADER_VOL',
                      'daysInYear': days_in_year,
                      'save': True,
                      'instruments': values,
                      'underlyer': {
                          'field': 'close',
                          'instance': 'CLOSE',
                          'quote': spot,
                          'instrumentId': underlying
                      }
                  },
                  'valuationDate': valuation_date,
                  'strikeType': 'PERCENT',
                  'tage': None,
                  'updatedAt': valuation_date,
                  'instrumentId': underlying,
                  'instance': 'CLOSE'
                  }

        model_information = _to_json(output)
        df = pd.DataFrame()
        df['instrument_id'] = [underlying]
        df['valuation_date'] = [valuation_date]
        df['model_information'] = [model_information]
        # df.to_csv('./surface.csv', columns=['instrument_id', 'valuation_date', 'model_information'], index=False)
        return df


def implied_vol_surface_calc(raw_data: pd.DataFrame, valuation_date: date, spot: float,
                             calendar: HolidayCalendar, days_in_year: int,
                             percent_strikes: List[float], business_days: List[int],
                             underlying: str = '000905.SH', use_qls_calibration: bool = True) -> Union[None, Tuple]:
    # data filtration
    logger.debug('start implied volatility data filtration')
    filtration_algo = ImpliedVolSurfaceAlgo(valuation_date=valuation_date,
                                            spot=spot,
                                            calendar=calendar)
    filtrated_data = filtration_algo.convert_market_vol_data(raw_data)
    if filtrated_data is None or len(filtrated_data.get('vols', [])) == 0:
        logger.warning('failed to get implied vols; use historical vol')
        return None
    logger.debug('data filtration is finished')

    # calibration process
    logger.debug('start svi params calibration')
    calibration_result = filtration_algo.get_calibrated_result(filtrated_data, use_qls_calibration=use_qls_calibration)
    logger.debug('calibration is finished')

    # get svi surface
    surface = calibration_result.get_calibrated_vol_surface().implied_vol_surface(percent_strikes, business_days)

    # save vol surface for display
    surface_df = ImpliedVolSurfaceAlgo.save_implied_vol_surface(vols=surface,
                                                                percent_strikes=percent_strikes,
                                                                spot=spot,
                                                                times_to_maturity=business_days,
                                                                days_in_year=250,
                                                                valuation_date=valuation_date)

    # save svi params for calculation
    svi_df = ImpliedVolSurfaceAlgo.save_svi_params(svi_params=calibration_result.params,
                                                   valuation_date=valuation_date,
                                                   days_in_year=days_in_year,
                                                   calendar=calendar,
                                                   underlying=underlying)
    return svi_df, surface_df
