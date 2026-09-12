"""Causal group-wise bias correction; residuals are against the frozen base forecast."""
from dataclasses import asdict, dataclass
import numpy as np
from q2_model import predict


@dataclass(frozen=True)
class BiasConfig:
    window_days: int = 7
    load_strength: float = 0.
    pv_strength: float = 0.

    def record(self):
        return asdict(self)


class BiasForecaster:
    """observe() is allowed only after forecast(), enforcing one-day causal updates."""
    def __init__(self, base_config):
        self.base_config=base_config
        self.history=[]
        self.residuals=[]
        self.residual_days=[]
        self.pending=None

    def forecast(self, config):
        if self.pending is not None:
            raise ValueError('Observe the pending day before requesting another forecast')
        n=len(self.history)
        base=np.zeros((144,2)) if n==0 else predict(self.history,self.base_config)
        correction=np.zeros_like(base)
        used=min(config.window_days,len(self.residuals))
        if used:
            # Six 4-hour groups; sample mean uses only previously completed days.
            errors=np.asarray(self.residuals[-used:])
            group_bias=errors.reshape(used,6,24,2).mean(axis=(0,2))
            correction=np.repeat(group_bias,24,axis=0)*np.array([config.load_strength,config.pv_strength])
        corrected=np.maximum(base+correction,0.)
        # A group correction must not manufacture nighttime PV from a zero baseline.
        corrected[base[:,1]==0,1]=0.
        self.pending=(n,base.copy())
        return corrected,dict(base=base.copy(),correction=corrected-base,
                              residual_days_used=used,
                              last_residual_day=self.residual_days[-1] if used else None)

    def observe(self,actual):
        if self.pending is None:
            raise ValueError('Issue a forecast before observing target-day actuals')
        n,base=self.pending
        actual=np.asarray(actual,dtype=float)
        if actual.shape!=(144,2) or not np.all(np.isfinite(actual)) or np.min(actual)<0:
            raise ValueError('Invalid observation')
        # Exclude cold-start and pre-week fallback residuals from bias calibration.
        if n>=7:
            self.residuals.append(actual.copy()-base)
            self.residual_days.append(n)
        self.history.append(actual.copy())
        self.pending=None


def candidates():
    yield BiasConfig(7,0.,0.)  # Explicit no-correction control; first in tie-break order.
    for window in (3,7):
        for load in (0.,.5,1.):
            for pv in (0.,.5,1.):
                if load==pv==0:continue
                yield BiasConfig(window,load,pv)
