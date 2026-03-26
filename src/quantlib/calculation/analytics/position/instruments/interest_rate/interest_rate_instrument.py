from abc import ABC


class IRInstrument(ABC):
    """
    abstract class for interest rate instrument
    """
    pass


class IRCashFlow(ABC):
    """
    abstract class for interest rate instrument's cash flow
    """
    pass


class FloatingLeg(IRCashFlow):
    pass


class FixedLeg(IRCashFlow):
    pass
