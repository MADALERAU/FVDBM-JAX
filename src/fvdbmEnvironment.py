import jax
from fvdbmElements import *

class Environment():
    def __init__(self):
        self.cells: Array[Cell] = []
        self.faces: Array[Face] = []