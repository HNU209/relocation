from typing import Union
import numpy as np

class Chromosome:
    def __init__(self, matrix: np.ndarray, fitness: Union[int, float] = 0) -> None:
        self.matrix = matrix
        self.fitness = fitness