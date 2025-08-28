# To make the SoccerEnv folder into a package
from .fieldconfig import FieldConfig
from .soccerenv import SoccerEnv

__all__ = ["FieldConfig", "SoccerEnv"]