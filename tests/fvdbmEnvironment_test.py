import jax

from src.fvdbmElements import *
from src.fvdbmDynamics import *
from src.fvdbmEnvironment import *

from utils.test_utils import *

def test_env_pmap(dynamics = D2Q9()):
    key = Key(1234)
    Element.dynamics = dynamics
    cells = [Cell.pdf_init(jax.random.uniform(key(),dynamics.ones_pdf().shape),[]) for i in range(10)]
    env = Environment(cells)
    env.calc_cell_eqs()

test_env_pmap()