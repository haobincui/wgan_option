from typing import List, Optional, Dict, Tuple

from quantlib.calculation.analytics.measure import RiskMeasure
from quantlib.calculation.analytics.position.pricer.pricer import BlackScholesValuationModelSingleAsset
from quantlib.calculation.analytics.position.pricer.scenario import BlackScholesScenarioDefinitionSingleAsset


def scenario_calc_blackscholes_sa(valuation_model: BlackScholesValuationModelSingleAsset,
                                  spot_changes: List[float],
                                  vol_changes: List[float],
                                  r_changes: List[float],
                                  q_changes: List[float],
                                  requests: Optional[List[RiskMeasure]] = None) \
        -> Dict[Tuple[int, int, int, int], dict]:
    scenarios = {}
    reqs = valuation_model.provides() if not requests else requests
    for spot_idx, dspot in enumerate(spot_changes):
        for vol_idx, dvol in enumerate(vol_changes):
            for r_idx, dr in enumerate(r_changes):
                for q_idx, dq in enumerate(q_changes):
                    scenario = BlackScholesScenarioDefinitionSingleAsset(spot_bump=dspot,
                                                                         vol_bump=dvol,
                                                                         r_bump=dr,
                                                                         q_bump=dq)
                    scenario_index = (spot_idx, vol_idx, r_idx, q_idx)
                    res = valuation_model.apply_scenario_adjustment(scenario).calc_many(reqs)
                    scenarios[scenario_index] = res
    return scenarios
