import jax

import sys
sys.path.append(".")
from src.elements import *
from src.dynamics import *
from src.environment import *

from utils.test_utils import *

def test_env_pmap(dynamics = D2Q9()):
    key = Key(1234)
    Element.dynamics = dynamics
    cells = [Cell.pdf_init(jax.random.uniform(key(),dynamics.ones_pdf().shape),[]) for i in range(10)]
    env = Environment.create(cells)
    params = env.init()
    env.calc_cell_eqs(params)

test_env_pmap()