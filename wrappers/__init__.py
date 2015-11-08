import sys
sys.path.append("./libs/rnade/buml")  # NOQA

from dbn import DBN
from va import VA
from nade import NADE
from simple_nade import SimpleNADE
from gsn import GSN
from adv import Adversarial
from bernoulli import BernoulliMixture
from rbm import RBM
from truth import Truth

list_models = [
    NADE, SimpleNADE,
    DBN, VA, GSN,
    Adversarial, BernoulliMixture,
    RBM,
    Truth
]
Models = dict((klass.__name__, klass)
              for klass in list_models)
