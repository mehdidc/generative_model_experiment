import sys
sys.path.append("./libs/rnade/buml")

from dbn import DBN
from va import VA
from nade import NADE
from simple_nade import SimpleNADE
from gsn import GSN
from adv import Adversarial
from bernoulli import BernoulliMixture
from rbm import RBM

Models = dict((klass.__name__, klass)
              for klass in [NADE, SimpleNADE, DBN, VA, GSN, Adversarial, BernoulliMixture, RBM])
