import sys
sys.path.append("./libs/rnade/buml")

from dbn import DBN
from va import VA
from nade import NADE
from gsn import GSN
from adv import Adversarial
from bernoulli import BernoulliMixture

Models = dict((klass.__name__, klass)
              for klass in [NADE, DBN, VA, GSN, Adversarial, BernoulliMixture])
