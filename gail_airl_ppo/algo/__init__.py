from .ppo import PPO, PPOAgent
from .sac import SAC, SACExpert
from .gail import GAIL
from .airl import AIRL, AIRLAgent
from .ppo_withrew import PPO_WITHREW, PPOAgent_WITHREW

from .odirl import ODIRL
from .odirl import ODIRLAgent

from .i2l import I2L

ALGOS = {
    'gail': GAIL,
    'airl': AIRL
}
