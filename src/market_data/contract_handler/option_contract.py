import re

import pandas as pd

from market_data.contract_handler.contract import Contract
from market_data.contract_handler.contract_type import ContractType
from quantlib.calculation.analytics.position.instruments.features import OptionType
from quantlib.calendar.utils import call_option_maturity_month_map, \
    put_option_maturity_month_map


class OptionContract(Contract):
    """
    Option contract handler class.
    e.g. FLG10000O0
    """

    def __init__(self, contract_name: str, contract_type: ContractType = ContractType.Option):
        super().__init__(contract_name, contract_type)
        if contract_type != ContractType.Option:
            raise ValueError(f'Contract Type: [{contract_type.name}] is not Option')
        self.contract_type = contract_type
        self.contract_name = contract_name

        contract_info = self.__get_option_contract_info()
        if contract_info is None:
            raise ValueError(f'Invalidate Contract Name: [{self.contract_name}]')
        self.__moneyness = int(contract_info['Moneyness'])
        self.__underlying = contract_info['Underlying']

        self.__maturity_year = int(contract_info['Year'])
        self.__maturity_month = contract_info['Month']

    def get_underlying(self) -> str:
        return self.__underlying

    def get_maturity_month_code(self) -> str:
        return self.__maturity_month

    def get_maturity_year_code(self) -> int:
        return self.__maturity_year

    def get_strike(self) -> float:
        return self.__get_strike_price()

    def get_option_type(self) -> OptionType:
        if self.__maturity_month in call_option_maturity_month_map.keys():
            return OptionType.CALL
        elif self.__maturity_month in put_option_maturity_month_map.keys():
            return OptionType.PUT
        else:
            raise ValueError(f'Invalidate Option Type: [{self.contract_name}]')

    def __get_option_contract_info(self):
        # pattern = r'(?P<Underlying>\D{2,3}(?=[a-zA-Z]{1}\d{1,2}))(?P<Month>[a-zA-Z]{1}(?=\d{1,2}))(?P<Year>\d{1,2})'
        # e.g. FLG12750N0; FLG10000O0
        pattern = r'(?P<Underlying>\D{2,3})(?P<Moneyness>\d{2,5})(?P<Month>[a-zA-Z]{1})(?P<Year>\d{1,2})'
        req = re.compile(pattern)

        res = req.search(self.contract_name)
        if res:
            matched_res = {
                'Contract Name': [f'{self.contract_name}'],
                **res.groupdict()
            }
        else:
            matched_res = None
            print(f'Not Found {self.contract_name}')

        return matched_res

    def __get_strike_price(self):
        if self.__underlying.upper() == 'TY':
            return self.__get_ty_strike_price()

        if self.__moneyness <= 100:  # 85
            return self.__moneyness
        if 100 < self.__moneyness < 5000:  # 1275, 1025, 850
            return self.__moneyness / 10

        if 5000 <= self.__moneyness < 10000:  # 8225
            return self.__moneyness / 100

        if self.__moneyness >= 10000:  # 10025
            return self.__moneyness / 100

    def __get_ty_strike_price(self) -> float:
        digits = len(str(self.__moneyness))

        if digits <= 2:
            return float(self.__moneyness)
        if digits == 3:
            if 100 <= self.__moneyness < 200:
                return float(self.__moneyness)
            return self.__moneyness / 10
        if digits == 4:
            if self.__moneyness < 5000:
                return self.__moneyness / 10
            return self.__moneyness / 100
        return self.__moneyness / 100

    def to_dict(self) -> dict:
        return {
            'contract_name': self.contract_name,
            'underlying': self.get_underlying(),
            'contract_type': self.contract_type.name,
            'strike': self.get_strike(),
            "option_type": self.get_option_type().name,
        }
