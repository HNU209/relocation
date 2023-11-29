from .chromosome import Chromosome
from typing import Union, Tuple
from itertools import product
from tqdm import tqdm
import pandas as pd
import numpy as np
import random

def init_population(init_state: list, matrix_space: Union[list, tuple], od_matrix: dict, population_size: int) -> list:
    pop = []
    for _ in tqdm(range(population_size)):
        matrix = np.zeros(matrix_space)
        
        for idx_i in range(matrix_space[0]):
            n_veh_per_grid = init_state[idx_i]
            valid_indices = [idx_j for idx_j in range(matrix_space[1]) if od_matrix[idx_i, idx_j]['time'] <= 10]
            
            if n_veh_per_grid > 0:
                if len(valid_indices) - 1 == 0:
                    matrix[idx_i, idx_i] = n_veh_per_grid
                elif n_veh_per_grid - 1 == 0:
                    matrix[idx_i, idx_i] = n_veh_per_grid
                else:
                    sample_size = min(len(valid_indices) - 1, n_veh_per_grid - 1)
                    random_points = np.random.choice(range(1, n_veh_per_grid), size=sample_size, replace=False)
                    random_points = np.append(random_points, [0, n_veh_per_grid])
                    random_points.sort()
                    assign_veh_n = np.diff(random_points)
                    
                    random_valid_indices = np.random.choice(valid_indices, size=len(assign_veh_n), replace=False)
                    for k, idx_j in enumerate(random_valid_indices):
                        matrix[idx_i, idx_j] = assign_veh_n[k]
        
        chromosome = Chromosome(matrix)
        chromosome = crossing_correction(chromosome)
        pop.append(chromosome)
    return pop

def _crossover(x: Chromosome, y: Chromosome) -> tuple:
    x_ = x.matrix.copy()
    y_ = y.matrix.copy()
    
    random_index = np.random.choice(x_.shape[0], size=int(x_.shape[0] // 2), replace=False)
    child1, child2 = y_.copy(), x_.copy()
    
    child1[random_index, :] = x_[random_index, :].copy()
    child2[random_index, :] = y_[random_index, :].copy()
    
    child1 = crossing_correction(Chromosome(child1))
    child2 = crossing_correction(Chromosome(child2))
    return child1, child2

def _mutate(x: Chromosome, mutation_rate: float, od_matrix: dict) -> Chromosome:
    if random.random() < mutation_rate:
        matrix_space = x.matrix.shape[0]
        random_row_index = np.random.randint(0, matrix_space)
        valid_indices = [idx_j for idx_j in range(matrix_space) if od_matrix[random_row_index, idx_j]['time'] <= 10]
        
        row_copy = x.matrix[random_row_index].copy()
        n_veh_per_grid = int(sum(row_copy))
        
        new_row = np.zeros(matrix_space)
        if n_veh_per_grid > 0:
            if len(valid_indices) - 1 == 0:
                new_row[random_row_index] = n_veh_per_grid
            elif n_veh_per_grid - 1 == 0:
                new_row[random_row_index] = n_veh_per_grid
            else:
                sample_size = min(len(valid_indices) - 1, n_veh_per_grid - 1)
                random_points = np.random.choice(range(1, n_veh_per_grid), size=sample_size, replace=False)
                random_points = np.append(random_points, [0, n_veh_per_grid])
                random_points.sort()
                
                assign_veh_n = np.diff(random_points)

                for k, idx_j in enumerate(valid_indices):
                    new_row[idx_j] = assign_veh_n[k] if k < len(assign_veh_n) else 0
        x.matrix[random_row_index] = new_row
        x = crossing_correction(x)
    return x

def calc_fitness(population: list, calling: list, arrival: list) -> Tuple[list, float]:
    avg_fit = 0
    for pop in population:
        result_matrix = np.zeros(len(pop.matrix[0]))
        for idx in range(len(pop.matrix[0])):
            curr_veh_n = pop.matrix[idx, :].sum()
            arrival_passen_n = arrival[idx]
            calling_passen_n = calling[idx]
            in_veh_n = pop.matrix[:, idx].sum()
            out_veh_n = pop.matrix[idx, :].sum()
            
            a_n = curr_veh_n + (arrival_passen_n - calling_passen_n + in_veh_n - out_veh_n)
            a_n = max(a_n, 1)
            result_matrix[idx] += (np.sqrt(2) / (3 * 15)) * np.sqrt(1 / a_n) * calling_passen_n
        
        fitness = result_matrix.sum()
        pop.fitness = fitness
        avg_fit += fitness
    avg_fit /= len(population)
    return population, avg_fit

def next_generation(population: list, population_size: int, mutation_rate: float, od_matrix: dict) -> list:
    new_pop = []
    while len(new_pop) < population_size:
        fitness_lst = np.array([1 / x.fitness for x in population])
        prob = fitness_lst / sum(fitness_lst)
        parents = np.random.choice(population, size=2, p=prob, replace=False)
        offspring_ = _crossover(parents[0], parents[1])
        child1 = _mutate(offspring_[0], mutation_rate, od_matrix)
        child2 = _mutate(offspring_[1], mutation_rate, od_matrix)
        offspring = [child1, child2]
        new_pop.extend(offspring)
    return new_pop

def crossing_correction(pop: Chromosome) -> Chromosome:
    matrix_len = pop.matrix.shape[0]
    
    for i in range(matrix_len):
        for j in range(i, matrix_len):
            grid_1 = pop.matrix[j, i]
            grid_2 = pop.matrix[i, j]
            
            if grid_1 == 0 or grid_2 == 0: continue
            else:
                min_vehs = min(grid_1, grid_2)
                ## 격자끼리 crossing 되는 차량 제외
                pop.matrix[j, i] -= min_vehs
                pop.matrix[i, j] -= min_vehs
                
                ## 제외된 차량은 기존 격자에서 대기
                pop.matrix[j, j] += min_vehs
                pop.matrix[i, i] += min_vehs
    return pop