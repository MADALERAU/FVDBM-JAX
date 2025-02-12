import jax

import sys
sys.path.append(".")
from src.elements import *
from src.dynamics import *
from src.environment import *

from utils.test_utils import *

def test_element_inits(dynamics = D2Q9()):
    print("*** Testing ELement Initializations ***")

    # Defining test vars
    key = Key(89524)    # Random key
    pdf = jax.random.uniform(key(),dynamics.ones_pdf().shape)
    rho = jax.random.uniform(key(),())
    vel = jax.random.uniform(key(),(2,))

    Element.dynamics = dynamics

    # Using Default Initialization
    el1 = Element(pdf,rho,vel)
    output = {"Random Init": el1,
              "PDF Size": el1.pdf.shape,
              "Rho Size": el1.rho.shape,
              "Vel Size": el1.vel.shape}
    print_dict(output)

    # Calculate Macro
    el1.calc_macro()
    output = {"Macro Calc": el1,
              "PDF Size": el1.pdf.shape,
              "Rho Size": el1.rho.shape,
              "Vel Size": el1.vel.shape}
    print_dict(output)

    # Init using PDF only
    el2 = Element.pdf_init(pdf)
    output = {"PDF Init": el2}
    print_dict(output)

    # Init using pdf of Ones
    el3 = Element.ones_init()
    output = {"Ones Init": el3}
    print_dict(output)

    print("*** End Test ***")

def test_cell_inits(dynamics=D2Q9()):
    print("*** Testing Cell Inits ***")
    
    # Defining test vars
    key = Key(582)
    pdf = jax.random.uniform(key(),dynamics.ones_pdf().shape)
    rho = jax.random.uniform(key(),())
    vel = jax.random.uniform(key(),(2,))
    pdf_eq = jax.random.uniform(key(),dynamics.ones_pdf().shape)
    face_index = (1)

    cell1 = Cell(pdf,rho,vel,pdf_eq,face_index)
    output = {"Random Init": cell1,
              "Non-Eq PDF": cell1.get_neq_pdf()}
    print_dict(output)
    cell2 = Cell.pdf_init(pdf,face_index)
    output = {"PDF Init": cell2}
    print_dict(output)


# Run Cases

test_element_inits()
test_cell_inits()