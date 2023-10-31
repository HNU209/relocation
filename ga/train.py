from tqdm import tqdm
from ga.utils import *
from env.env import Env
import time

def ga_runner(args: dict, env: Env) -> None:
    mutation_rate = args['mutation_rate']
    num_generation = args['num_generation']
    population_size = args['population_size']
    relocation_timing = args['relocation_timing']
    
    start_time = args['start_time']
    end_time = args['end_time']
    
    env.reset()
    matrix_space = env.get_matrix_space()
    
    total_running_time = time.time()
    for t in range(start_time, end_time):
        env.assign_veh_to_passenger()
        
        env.go_to_destination()
        
        env._update_after_running()
        
        ### relocation
        s_t = time.time()
        if t != 0 and t % relocation_timing == 0:
            veh_state, calling_passen_state, arrival_passen_state = env.get_state()
            pop = init_population(veh_state, matrix_space, env.od_matrix, population_size)
            
            best_pop = None
            
            for gen in tqdm(range(1, num_generation + 1)):
                pop, avg_fit = calc_fitness(pop, calling_passen_state, arrival_passen_state)
                new_pop = next_generation(pop, population_size, mutation_rate, env.od_matrix)
                best_pop_ = list(sorted(pop, key=lambda x: x.fitness, reverse=True))[-1]
                
                if best_pop == None or best_pop.fitness > best_pop_.fitness:
                    best_pop = best_pop_
                
                pop = new_pop
                
                if gen % 10 == 0:
                    print(f'Generation : {gen} - avg_fit : {avg_fit:.4f}')
            env.ga_step(best_pop)
        e_t = time.time()
        print(t, e_t - s_t)
        env.step()
        
        if args['verbose']:
            print(f'\nCurrent Time : {t}')
            total_vehs = sum(list(map(lambda x: list(x), env.curr_vehs.values())), [])
            total_passengers = sum(list(map(lambda x: list(x), env.curr_passengers.values())), [])
            print(f'free vehs : {len(list(filter(lambda x: x.veh_state == "free", total_vehs)))}')
            print(f'not free vehs : {len(list(filter(lambda x: x.veh_state == "not free", total_vehs)))}')
            print(f'assigned vehs : {len(list(filter(lambda x: x.veh_state == "assigned", total_vehs)))}')
            print(f'move vehs : {len(list(filter(lambda x: x.veh_state == "move", total_vehs)))}')
            print(f'calling passengers : {len(list(filter(lambda x: x.passenger_state == "call", total_passengers)))}')
            print(f'arrived passengers : {len(list(filter(lambda x: x.passenger_state == "arrive", total_passengers)))}\n')

    env._save_to_json()
    total_running_time = time.time() - total_running_time
    print(f'Total running time - {int(total_running_time // 60):2d}min : {round(total_running_time % 60):02d}sec')