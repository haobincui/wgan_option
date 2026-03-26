from dataclasses import dataclass


@dataclass(frozen=True)
class Currency:
    name: str

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'Currency(name={self.name})'


CNY = Currency(name='CNY')
HKD = Currency(name='HKD')
USD = Currency(name='USD')
EUR = Currency(name='EUR')
MXN = Currency(name='MXN')
ARS = Currency(name='ARS')
CLP = Currency(name='CLP')
BRL = Currency(name='BRL')

domestic_currency = CNY


@dataclass(frozen=True)
class CurrencyPair:
    base: Currency
    quote: Currency

    @property
    def name(self) -> str:
        return self.base.name + self.quote.name

    def flip(self):
        return CurrencyPair(base=self.quote, quote=self.base)

    def __str__(self):
        return f'{self.base.name}{self.quote.name}'

    def __repr__(self):
        return f'CurrencyPair({self.base.__repr__()}, {self.quote.__repr__()})'


USD_CNY = CurrencyPair(USD, CNY)
EUR_USD = CurrencyPair(EUR, USD)
CNY_HKD = CurrencyPair(CNY, HKD)
