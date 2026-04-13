from abc import ABC, abstractmethod
from dataclasses import replace
from datetime import date, time
from typing import List, Optional, Any, Dict

from quantlib.calculation.analytics.calculators.equity_calculator import blackscholes_const_params_pv_analytic
from quantlib.calculation.analytics.measure import RiskMeasure
from quantlib.calculation.analytics.models.utils import double_is_zero
from quantlib.calculation.analytics.position.instruments.features import Instrument
from quantlib.calculation.analytics.position.instruments.interest_rate.european import VanillaEuropean
from quantlib.calculation.analytics.position.pricer.config import BlackScholesPricerConfig, blackscholes_default_config
from quantlib.calculation.analytics.position.pricer.scenario import ScenarioDefinition, \
    BlackScholesScenarioDefinitionSingleAsset
from quantlib.calendar.schedule import BusinessDayConvention, plus_period, EomConvention


class ValuationModel(ABC):
    """
    A pricer is a combination of an instrument, market data, model and model parameters
    such that one can query the pricer for the instrument's pv, greeks etc. Pv is the most
    basic requirement of a pricer. A pricing algorithm may be able to calculate more than pv.
    For example, a PDE solver may calculate pv, delta and gamma at the same time.
    """

    @abstractmethod
    def provides(self) -> List[RiskMeasure]:
        """
        Lists the pricer's capabilities
        :return: Pricer measures that the pricer supports
        """
        pass

    @abstractmethod
    def pv(self) -> float:
        """
        The most basic interface/capability of a pricer: PV calculation
        :return: Instrument pv
        """
        pass

    @abstractmethod
    def apply_scenario_adjustment(self, scenario_definition: ScenarioDefinition):
        pass

    @abstractmethod
    def get_valuation_date(self) -> date:
        pass

    def get_valuation_time(self) -> Optional[time]:
        return None

    def calc(self, request: RiskMeasure) -> Any:
        """
        Compute a specific measure
        :param request: The measure to compute
        :return: Computed measure. WARNING: only PV is ensured. All other requests throw.
        """
        if request == RiskMeasure.PV:
            return self.pv()
        else:
            raise ValueError(f'Pricer::calc: requested measure {request.name} not supported')

    def calc_many(self, requests: List[RiskMeasure]) -> Dict[RiskMeasure, Any]:
        """
        Compute a select of measures
        :param requests: Measures to compute
        :return: Computed measures
        """
        return {r: self.calc(r) for r in requests}

    def calc_all(self) -> Dict[RiskMeasure, Any]:
        """
        Compute all supported measures
        :return: Computed measures
        """
        return self.calc_many(self.provides())


class BlackScholesValuationModelSingleAsset(ValuationModel):
    """
    Single asset Black-scholes pricer based on lognormal models with constant vol, r, q parameters.
    The only model parameters are underlyer spot, vol, r and q. These parameters are
    floating numbers. For the same instrument and same model parameters, there are
    still various numerical methods to trade_price the instrument.
    """

    def __init__(self,
                 instrument: Instrument,
                 spot: float,
                 vol: float,
                 r: float,
                 q: float,
                 valuation_date: date,
                 config: BlackScholesPricerConfig,
                 quantity: float,
                 valuation_time: Optional[time]):
        self.instrument = instrument
        self.spot = spot
        self.vol = vol
        self.r = r
        self.q = q
        self.valuation_date = valuation_date
        self.valuation_time = valuation_time
        self.config = config
        self.quantity = quantity
        try:
            self.tau = config.vol_calendar(valuation_date, instrument.expiration_date)
        except Exception as e:
            self.tau = 0

    @abstractmethod
    def pv(self) -> float:
        pass

    def calc(self, request: RiskMeasure) -> Any:
        return self.calc_many([request])[request]

    def calc_many(self, requests: List[RiskMeasure]) -> Dict[RiskMeasure, Any]:
        pv = 0
        sp = 0
        sm = 0
        vp = 0
        vm = 0
        rp = 0
        qp = 0
        qm = 0
        rm = 0
        tp = 0
        kp = 0
        km = 0
        dspot = self.config.dspot_percent * self.spot if not double_is_zero(self.spot) else 0.01
        vol_bump = self.config.dvol
        #  for cash or forward, vol can be 0. so for fd vega, we cannot use 1% * vol which is 0.
        if not double_is_zero(self.vol) and vol_bump >= self.vol:
            vol_bump = 0.01 * self.vol
        nvd = None

        if RiskMeasure.PV in requests \
                or RiskMeasure.THETA in requests \
                or RiskMeasure.GAMMA in requests \
                or RiskMeasure.DENSITY in requests:
            pv = self.pv()
        if RiskMeasure.DELTA in requests or RiskMeasure.GAMMA in requests:
            spot_up = BlackScholesScenarioDefinitionSingleAsset(spot_bump=dspot)
            sp = self.apply_scenario_adjustment(spot_up).pv()
            spot_down = BlackScholesScenarioDefinitionSingleAsset(spot_bump=-dspot)
            sm = self.apply_scenario_adjustment(spot_down).pv()
        if RiskMeasure.VEGA in requests:
            vol_up = BlackScholesScenarioDefinitionSingleAsset(vol_bump=vol_bump)
            vp = self.apply_scenario_adjustment(vol_up).pv()
            vol_down = BlackScholesScenarioDefinitionSingleAsset(vol_bump=-vol_bump)
            vm = self.apply_scenario_adjustment(vol_down).pv()
        if RiskMeasure.RHO in requests:
            r_up = BlackScholesScenarioDefinitionSingleAsset(r_bump=self.config.dr)
            rp = self.apply_scenario_adjustment(r_up).pv()
            r_down = BlackScholesScenarioDefinitionSingleAsset(r_bump=-self.config.dr)
            rm = self.apply_scenario_adjustment(r_down).pv()
        if RiskMeasure.RHO_Q in requests:
            q_up = BlackScholesScenarioDefinitionSingleAsset(q_bump=self.config.dq)
            qp = self.apply_scenario_adjustment(q_up).pv()
            q_down = BlackScholesScenarioDefinitionSingleAsset(q_bump=-self.config.dq)
            qm = self.apply_scenario_adjustment(q_down).pv()
        if RiskMeasure.THETA in requests:
            shift_tomorrow = BlackScholesScenarioDefinitionSingleAsset(
                valuation_date_shift=self.config.next_day)
            shifted_pricer = self.apply_scenario_adjustment(shift_tomorrow)
            nvd = shifted_pricer.get_valuation_date()
            tp = shifted_pricer.pv()
        if RiskMeasure.DENSITY in requests:
            k_up = BlackScholesScenarioDefinitionSingleAsset(strike_bump=self.config.dk)
            kp = self.apply_scenario_adjustment(k_up).pv()
            k_down = BlackScholesScenarioDefinitionSingleAsset(strike_bump=-self.config.dk)
            km = self.apply_scenario_adjustment(k_down).pv()

        result = {}
        if RiskMeasure.PV in requests:
            result[RiskMeasure.PV] = pv
        if RiskMeasure.DELTA in requests:
            result[RiskMeasure.DELTA] = (sp - sm) * 0.5 / dspot
        if RiskMeasure.GAMMA in requests:
            result[RiskMeasure.GAMMA] = (sp - 2 * pv + sm) / (dspot * dspot)
        if RiskMeasure.VEGA in requests:
            result[RiskMeasure.VEGA] = (vp - vm) * 0.5 / vol_bump
        if RiskMeasure.RHO in requests:
            result[RiskMeasure.RHO] = (rp - rm) * 0.5 / self.config.dr
        if RiskMeasure.RHO_Q in requests:
            result[RiskMeasure.RHO_Q] = (qp - qm) * 0.5 / self.config.dq
        if RiskMeasure.THETA in requests:
            dtau = self.config.vol_calendar(self.valuation_date, nvd)
            result[RiskMeasure.THETA] = (tp - pv) / dtau
        if RiskMeasure.DENSITY in requests:
            result[RiskMeasure.DENSITY] = (kp - 2 * pv + km) / (self.config.dk * self.config.dk)
        return result

    def calc_all(self) -> Dict[RiskMeasure, Any]:
        return self.calc_many(self.provides())

    def provides(self) -> List[RiskMeasure]:
        return [RiskMeasure.PV, RiskMeasure.DELTA, RiskMeasure.GAMMA,
                RiskMeasure.VEGA, RiskMeasure.THETA, RiskMeasure.RHO, RiskMeasure.RHO_Q]

    def get_valuation_date(self) -> date:
        return self.valuation_date

    def get_valuation_time(self) -> Optional[time]:
        return self.valuation_time

    def apply_scenario_adjustment(self, scenario_definition: ScenarioDefinition) -> ValuationModel:
        if not isinstance(scenario_definition, BlackScholesScenarioDefinitionSingleAsset):
            raise ValueError(f'apply_scenario_adjustment: scenario is not of type BlackScholesScenarioDefinition')
        new_valuation_date = plus_period(self.valuation_date,
                                         scenario_definition.valuation_date_shift,
                                         adj=BusinessDayConvention.FOLLOWING,
                                         calendar=self.config.calendar,
                                         eom=EomConvention.NONE)

        if isinstance(self.instrument, VanillaEuropean):
            new_k = self.instrument.strike + scenario_definition.strike_bump
            new_instrument = VanillaEuropean(
                strike=new_k,
                expiration_date=self.instrument.expiration_date,
                option_type=self.instrument.option_type,
                delivery_date=self.instrument.delivery_date,
                underlying=self.instrument.underlying
            )
        else:
            new_instrument = self.instrument

        return self.__class__(
            # instrument=roll_instrument(self.instrument, self.spot, self.valuation_date, new_valuation_date),
            instrument=new_instrument,
            spot=self.spot + scenario_definition.spot_bump,
            vol=self.vol + scenario_definition.vol_bump,
            r=self.r + scenario_definition.r_bump,
            q=self.q + scenario_definition.q_bump,
            valuation_date=new_valuation_date,
            config=self.config,
            quantity=self.quantity,
            valuation_time=self.valuation_time)


class BlackScholesValuationModelSingleAssetAnalytic(BlackScholesValuationModelSingleAsset):

    def __init__(self,
                 instrument: Instrument,
                 spot: float,
                 vol: float,
                 r: float,
                 q: float,
                 valuation_date: date,
                 config: BlackScholesPricerConfig = blackscholes_default_config,
                 quantity: float = 1,
                 valuation_time: Optional[time] = None):
        super().__init__(instrument, spot, vol, r, q, valuation_date, config, quantity, valuation_time)

    def pv(self) -> float:
        #  try to handle the case when the option has expired
        #  don't know if expiration_date exists. so put in a try block.
        # try:
        #     if self.instrument.expiration_date <= self.valuation_date:
        #         return self.quantity * immediate_exercise(self.instrument, self.spot, self.valuation_date)
        # except:
        #     pass
        return self.quantity * blackscholes_const_params_pv_analytic(
            self.instrument,
            self.spot,
            self.vol,
            self.r,
            self.q,
            self.valuation_date,
            self.config)


class Black76ValuationModelSingleAsset(ValuationModel):
    """
    Black76 is a specialization of Black-Scholes model with q set to be equal to r.
    So a Black76 pricer is simply a wrapper around a BlackScholes pricer.
    """

    def __init__(self, blackscholes_pricer: BlackScholesValuationModelSingleAsset):
        if not double_is_zero(blackscholes_pricer.r - blackscholes_pricer.q):
            raise ValueError('black-scholes pricer r and q must be equal when constructing a black76 pricer out of it')
        self.blackscholes_pricer = blackscholes_pricer
        self.spot = blackscholes_pricer.spot
        self.vol = blackscholes_pricer.vol
        self.r = blackscholes_pricer.r
        self.q = blackscholes_pricer.r
        self.tau = blackscholes_pricer.tau

    def pv(self) -> float:
        return self.blackscholes_pricer.pv()

    def calc(self, request: RiskMeasure) -> Any:
        # special handling of rho_q
        if request == RiskMeasure.RHO:
            dr = self.blackscholes_pricer.config.dr
            r_up = BlackScholesScenarioDefinitionSingleAsset(r_bump=dr, q_bump=dr)
            qp = self.blackscholes_pricer.apply_scenario_adjustment(r_up).pv()
            r_down = BlackScholesScenarioDefinitionSingleAsset(r_bump=-dr, q_bump=-dr)
            qm = self.blackscholes_pricer.apply_scenario_adjustment(r_down).pv()
            return (qp - qm) * 0.5 / dr
        return self.blackscholes_pricer.calc(request)

    def calc_many(self, requests: List[RiskMeasure]) -> Dict[RiskMeasure, Any]:
        return self.blackscholes_pricer.calc_many(requests)

    def calc_all(self) -> Dict[RiskMeasure, Any]:
        return self.calc_many(self.provides())

    def provides(self) -> List[RiskMeasure]:
        return [RiskMeasure.PV, RiskMeasure.DELTA, RiskMeasure.GAMMA,
                RiskMeasure.VEGA, RiskMeasure.THETA, RiskMeasure.RHO]

    def get_valuation_date(self) -> date:
        return self.blackscholes_pricer.get_valuation_date()

    def get_valuation_time(self) -> Optional[time]:
        return self.blackscholes_pricer.get_valuation_time()

    def apply_scenario_adjustment(self, scenario_definition: ScenarioDefinition) -> ValuationModel:
        if not isinstance(scenario_definition, BlackScholesScenarioDefinitionSingleAsset):
            raise ValueError(f'apply_scenario_adjustment: scenario is not of type BlackScholesScenarioDefinition')
        adjusted = replace(scenario_definition, q_bump=scenario_definition.r_bump)
        return Black76ValuationModelSingleAsset(self.blackscholes_pricer.apply_scenario_adjustment(adjusted))

# class BlackScholesValuationModelSingleAssetForeign(ValuationModel):
#     """
#     Foreign underlyer with foreign payoff. Optionally converted to domestic with floating FX rate.
#     for example, strike 100USD call. At expiry, underlyer trade_price is 105USD. The payoff is 5USD.
#     If upon expiry the FX rate is 1USD = 7CNY, then domestic payoff is 5 * 7 CNY.
#     This pricer is really a very simple transformation (only multiplying final results by fx rate) of
#     a single asset pricer.
#     """
#
#     def __init__(self,
#                  pricer: BlackScholesValuationModelSingleAsset,
#                  fx_spot: float):
#         self.pricer = pricer
#         self.fx_spot = fx_spot
#
#     def pv(self) -> float:
#         return self.pricer.pv() * self.fx_spot
#
#     def calc_many(self, requests: List[RiskMeasure]) -> Dict[RiskMeasure, Any]:
#         reqs = [r for r in requests if not r == RiskMeasure.FX_DELTA]
#         res = self.pricer.calc_many(reqs)
#         res = {k: v * self.fx_spot for k, v in res.items()}
#         if RiskMeasure.FX_DELTA in requests:
#             if RiskMeasure.PV in requests:
#                 res[RiskMeasure.FX_DELTA] = res[RiskMeasure.PV]
#             else:
#                 res[RiskMeasure.FX_DELTA] = self.pricer.pv()
#         return res
#
#     def provides(self) -> List[RiskMeasure]:
#         return [RiskMeasure.PV, RiskMeasure.DELTA, RiskMeasure.GAMMA,
#                 RiskMeasure.VEGA, RiskMeasure.THETA, RiskMeasure.RHO, RiskMeasure.RHO_Q,
#                 RiskMeasure.FX_DELTA]
#
#     def apply_scenario_adjustment(self, scenario_definition: ScenarioDefinition) -> ValuationModel:
#         if not isinstance(scenario_definition, BlackScholesScenarioDefinitionSingleAssetForeign):
#             raise ValueError('foreign option pricer requires BlackScholesScenarioDefinitionSingleAssetForeign')
#         return BlackScholesValuationModelSingleAssetForeign(self.pricer.
#                                                             apply_scenario_adjustment(scenario_definition.
#                                                                                       to_black_scholes_scenario_sa()),
#                                                             self.fx_spot + scenario_definition.fx_spot_bump)
#
#     def get_valuation_date(self) -> date:
#         return self.pricer.get_valuation_date()
