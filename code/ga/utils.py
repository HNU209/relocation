from .chromosome import Chromosome
from itertools import product
import pandas as pd
import numpy as np
import random

def init_population(init_state, matrix_space, od_matrix, population_size):
    pop = []
    for _ in range(population_size):
        matrix = np.zeros(matrix_space)
        for idx_i in range(matrix_space[0]):
            n_veh_per_grid = init_state[idx_i]
            for idx_j in range(matrix_space[1]):
                if od_matrix[idx_i, idx_j]['time'] > 10:
                    assign_veh_n = 0
                else:
                    if n_veh_per_grid == 0:
                        assign_veh_n = 0
                    else:
                        assign_veh_n = random.randint(0, n_veh_per_grid)
                matrix[idx_i, idx_j] = assign_veh_n
                n_veh_per_grid -= assign_veh_n
            
            if n_veh_per_grid != 0:
                raise
        
        chromosome = Chromosome(matrix)
        pop.append(chromosome)
    return pop

def _crossover(x, y):
    x_ = x.matrix.copy()
    y_ = y.matrix.copy()
    
    random_index = np.random.choice(x_.shape[0], size=int(x_.shape[0] // 2), replace=False)
    child1, child2 = y_.copy(), x_.copy()
    
    child1[random_index, :] = x_[random_index, :].copy()
    child2[random_index, :] = y_[random_index, :].copy()
    
    child1 = Chromosome(child1)
    child2 = Chromosome(child2)
    return child1, child2

def _mutate(x: Chromosome, mutation_rate: float) -> Chromosome:
    if random.random() < mutation_rate:
        random_row_index = np.random.randint(0, x.matrix.shape[0])
        
        row_copy = x.matrix[random_row_index].copy()
        row_n_vehs = int(sum(row_copy))
        row_n_vehs_copy = int(sum(row_copy))
        new_row = []
        
        while len(new_row) < len(row_copy):
            if row_n_vehs == 0:
                n_veh = 0
            else:
                n_veh = random.choice(range(row_n_vehs_copy))
                row_n_vehs_copy -= n_veh
            new_row.append(n_veh)
        
        if sum(new_row) != row_n_vehs:
            for _ in range(row_n_vehs - sum(new_row)):
                random_index = np.random.choice(len(new_row), size=1, replace=False)[0]
                new_row[random_index] += 1
        x.matrix[random_row_index] = np.array(new_row)
    return x

def calc_fitness(population, calling, arrival):
    avg_fit = 0
    for pop in population:
        result_matrix = np.zeros(len(pop.matrix[0]))
        for idx in range(len(pop.matrix[0])):
            curr_veh_n = pop.matrix[idx, :].sum()
            arrival_passen_n = arrival[idx]
            calling_passen_n = calling[idx]
            in_veh_n = pop.matrix[: idx].sum() - pop.matrix[idx, idx]
            out_veh_n = pop.matrix[idx, :].sum() - pop.matrix[idx, idx]
            
            a_n = curr_veh_n + (arrival_passen_n - calling_passen_n + in_veh_n - out_veh_n)
            a_n = max(a_n, 1)
            result_matrix[idx] += (np.sqrt(2) / (3 * 15)) * np.sqrt(1 / a_n) * calling_passen_n
        
        fitness = result_matrix.sum()
        pop.fitness = fitness
        avg_fit += fitness
    avg_fit / len(population)
    return population, avg_fit

def next_generation(population, population_size, mutation_rate, check):
    new_pop = []
    while len(new_pop) < population_size:
        fitness_lst = np.array([1 / x.fitness for x in population])
        prob = fitness_lst / sum(fitness_lst)
        parents = np.random.choice(population, size=2, p=prob, replace=False)
        offspring_ = _crossover(parents[0], parents[1])
        child1 = _mutate(offspring_[0], mutation_rate)
        child2 = _mutate(offspring_[1], mutation_rate)
        offspring = [child1, child2]
        new_pop.extend(offspring)
    return new_pop

### valid matrix function
# def _check_intersect_vehs(matrix):
#     new_matrix = matrix.copy()
#     for i in range(len(matrix[0])):
#         for j in range(i + 1, len(matrix[1])):
#             if matrix[i, j] > matrix[j, i]:
#                 diff_veh_n = matrix[i, j] - matrix[j, i]
                
#                 new_matrix[i, i] += matrix[i, j] - diff_veh_n
#                 new_matrix[j, j] += matrix[j, i]
                
#                 new_matrix[i, j] = diff_veh_n
#                 new_matrix[j, i] = 0
                
#             elif matrix[i, j] == matrix[j, i]:
#                 new_matrix[i, i] += matrix[i, j]
#                 new_matrix[j, j] += matrix[j, i]
                
#                 new_matrix[i, j] = 0
#                 new_matrix[j, i] = 0
                
#             elif matrix[i, j] < matrix[j, i]:
#                 diff_veh_n = matrix[j, i] - matrix[i, j]
                
#                 new_matrix[i, i] += matrix[i, j]
#                 new_matrix[j, j] += matrix[j, i] - diff_veh_n
                
#                 new_matrix[i, j] = 0
#                 new_matrix[j, i] = diff_veh_n
                
#     return new_matrix