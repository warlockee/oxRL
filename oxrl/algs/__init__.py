from oxrl.algs.sft import SFT
from oxrl.algs.dpo import DPO
from oxrl.algs.orpo import ORPO
from oxrl.algs.kto import KTO
from oxrl.algs.cpt import CPT
from oxrl.algs.kd import KD
from oxrl.algs.rm import RM
from oxrl.algs.online_dpo import OnlineDPO
from oxrl.algs.rft import RFT
from oxrl.algs.spin import SPIN
from oxrl.algs.ipo import IPO
from oxrl.algs.simpo import SimPO
from oxrl.algs.cpo import CPO
from oxrl.algs.alphapo import AlphaPO
from oxrl.algs.rdpo import RDPO
from oxrl.algs.cdpo import CDPO
from oxrl.algs.betadpo import BetaDPO
from oxrl.algs.caldpo import CalDPO
from oxrl.algs.sppo import SPPO
from oxrl.algs.aot import AOT
from oxrl.algs.apo import APO
from oxrl.algs.nca import NCA
from oxrl.algs.hinge import Hinge
from oxrl.algs.robust_dpo import RobustDPO
from oxrl.algs.exo import EXO
from oxrl.algs.discopop import DiscoPOP
from oxrl.algs.bco import BCO
from oxrl.algs.odpo import ODPO
from oxrl.algs.dpop import DPOP
from oxrl.algs.focalpo import FocalPO
from oxrl.algs.gpo import GPO
from oxrl.algs.wpo import WPO
from oxrl.algs.fdpo import FDPO
from oxrl.algs.hdpo import HDPO
from oxrl.algs.dposhift import DPOShift
from oxrl.algs.cposimpo import CPOSimPO
from oxrl.algs.sampo import SamPO
from oxrl.algs.drdpo import DrDPO
from oxrl.algs.chipo import ChiPO
from oxrl.algs.spo import SPO
from oxrl.algs.dpnll import DPNLL
from oxrl.algs.minor_dpo import MinorDPO
from oxrl.algs.c2dpo import C2DPO
from oxrl.algs.alpha_dpo import AlphaDPO as AlphaDPOMethod
from oxrl.algs.bpo import BPO

# RL algorithms require ray/vllm — import lazily to avoid hard dependency
def _lazy_import_grpo():
    from oxrl.algs.grpo import GRPO
    return GRPO

def _lazy_import_ppo():
    from oxrl.algs.ppo import PPO
    return PPO

__all__ = [
    "SFT", "DPO", "ORPO", "KTO",
    "CPT", "KD", "RM", "OnlineDPO", "RFT", "SPIN",
    "IPO", "SimPO", "CPO", "AlphaPO", "RDPO", "CDPO",
    "BetaDPO", "CalDPO", "SPPO", "AOT", "APO", "NCA", "Hinge",
    "RobustDPO", "EXO", "DiscoPOP", "BCO", "ODPO", "DPOP", "FocalPO",
    "GPO", "WPO", "FDPO", "HDPO", "DPOShift", "CPOSimPO", "SamPO",
    "DrDPO", "ChiPO", "SPO", "DPNLL", "MinorDPO", "C2DPO", "AlphaDPOMethod", "BPO",
]
