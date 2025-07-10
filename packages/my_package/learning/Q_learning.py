import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys

if "../" not in sys.path:
  sys.path.append("../") 

from collections import defaultdict
from learning import DukietownEnv

matplotlib.style.use('ggplot')

env = DukietownEnv()

