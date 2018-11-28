import platform
from .spec import Space, Spec
from .abc import Env
from .shm_multiproc import ShmMultiProcEnv
from .msg_multiproc import MsgMultiProcEnv

MultiProcEnv = ShmMultiProcEnv
if platform.system() == 'Windows':
    MultiProcEnv = MsgMultiProcEnv
