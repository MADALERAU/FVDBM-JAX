from src.fvdbmDynamics import D2Q9
import jax
import timeit

key = jax.random.key(1)

tempD2Q9 = D2Q9()


x = tempD2Q9.ones_pdf()
rho = jax.random.uniform(key,(1,))
vel = jax.random.uniform(key,(2,))


print(timeit.timeit(setup='''
from src.fvdbmDynamics import D2Q9
import jax
key = jax.random.key(1)
tempD2Q9 = D2Q9()
x = tempD2Q9.ones_pdf()
rho = jax.random.uniform(key,(1,))
vel = jax.random.uniform(key,(2,))''',
stmt='tempD2Q9.calc_eq(rho,vel)',number=1000))
