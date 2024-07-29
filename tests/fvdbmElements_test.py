from src.fvdbmElements import *
from src.fvdbmDynamics import *
from src.fvdbmEnvironment import *

Element.dynamics = D2Q9()
testElement = Element.ones_init()
print(testElement.density())
testElement.pdf = jnp.zeros_like(testElement.pdf)
print(testElement.density())

testCell = Cell(testElement.pdf,testElement.rho,testElement.vel,testElement.pdf,faces_index=[])
print(testCell.get_neq_pdf())

testCell2 = Cell.eq_init(testElement.pdf,testElement.rho,testElement.vel,faces_index=[])
print(testCell.pdf_eq)

cells = [Cell(testElement.pdf,testElement.rho,testElement.vel,testElement.pdf,faces_index=[]) for i in range(10)]
testEnv = Environment(cells=cells)
print(jax.tree.structure(testEnv))