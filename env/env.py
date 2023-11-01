from ortools.linear_solver import pywraplp
from ga.chromosome import Chromosome
from typing import Tuple
import geopandas as gpd
import pandas as pd
import json
import sys

try:
    from tools import *
    from object import *
    from tools import _haversine
except:
    sys.path.append('/home/happy956/relocation')
    sys.path.append('/home/happy956/relocation/env')
    from tools import *
    from object import *
    from tools import _haversine

class Env:
    def __init__(self, args: dict) -> None:
        self.args = args
        self.place_name = self.args['place_name']
        self.unit_size = self.args['unit_size']
        self.unit = self.args['unit']
        
        self.n_taxi_point = self.args['n_taxi_point']
        self.n_passenger_point = self.args['n_passenger_point']
        
        self.generate_data = self.args['generate_data']
        self.use_random_data = self.args['use_random_data']
        
        self._setup()
        
        self.current_time = 0
        self.is_done = False
        
        ### save data
        self.vehicle_point_data = []
        self.passenger_point_data = []
        self.trip_data = []
        
        self.reset()
    
    def get_matrix_space(self) -> tuple:
        return (len(self.grids), len(self.grids))
    
    def reset(self) -> None:
        self.current_time = 0
        self.is_done = False
        
        self.curr_vehs = {grid_id:[] for grid_id in self.grids.keys()}
        self.curr_passengers = {grid_id:[] for grid_id in self.grids.keys()}
        self.predict_calling_passenger = {grid_id:0 for grid_id in self.grids.keys()}
        self.predict_arrival_passenger = {grid_id:0 for grid_id in self.grids.keys()}
        
        for veh in self.total_vehs:
            if veh.start_time <= self.current_time:
                self.curr_vehs[veh.init_grid].append(veh)
        
        for passenger in self.total_passengers:
            if passenger.start_time <= self.current_time:
                self.curr_passengers[passenger.init_ori_grid].append(passenger)

            if passenger.start_time <= self.current_time + 30:
                self.predict_calling_passenger[passenger.init_ori_grid] += 1
    
    def get_state(self) -> Tuple[list, list, list]:
        sorted_veh_dict = dict(sorted(self.curr_vehs.items(), key=lambda x: x[0]))
        sorted_veh_lst = [len(list(filter(lambda x: x.veh_state == 'free' and x.delay == 0, veh_lst))) for veh_lst in sorted_veh_dict.values()]
        
        ### (2023-10-20 : 예측 모형 X)
        predict_calling_passengers_dict = dict(sorted(self.predict_calling_passenger.items(), key=lambda x: x[0]))
        predict_arrival_passengers_dict = dict(sorted(self.predict_arrival_passenger.items(), key=lambda x: x[0]))
        
        predict_calling_passengers_lst = list(predict_calling_passengers_dict.values())
        predict_arrival_passengers_lst = list(predict_arrival_passengers_dict.values())
        return sorted_veh_lst, predict_calling_passengers_lst, predict_arrival_passengers_lst
    
    def ga_step(self, chromosome: Chromosome) -> None:
        for from_grid_id, row in enumerate(chromosome.matrix):
            if sum(row) == 0: continue
            
            for to_grid_id, col in enumerate(row):
                if from_grid_id == to_grid_id or col == 0:
                    continue
                
                od_data = self.od_matrix[from_grid_id, to_grid_id]
                avail_veh_lst = list(filter(lambda x: x.veh_state == 'free' and x.delay == 0, self.curr_vehs[from_grid_id]))
                random_veh_lst = np.random.choice(avail_veh_lst, size=int(col), replace=False)
                
                for idx, veh in enumerate(random_veh_lst):
                    poped_veh_id = [idx_ for idx_, veh_ in enumerate(self.curr_vehs[from_grid_id]) if veh_.veh_id == veh.veh_id][0]
                    veh = self.curr_vehs[from_grid_id].pop(poped_veh_id)
                    
                    random_dest_index = np.random.choice(5, 1)[0]
                    random_dest = od_data['random_points_d'][f'{random_dest_index}']
                    lat_d, lon_d = random_dest['lat'], random_dest['lon']
                    
                    lat1, lon1, lat2, lon2 = veh.curr_lat, veh.curr_lon, lat_d, lon_d
                    routes, distance, duration = get_duration_distance((lon1, lat1, lon2, lat2), route=True)
                    if len(routes) == 0: raise
                    timestamp = get_timestamp(routes, self.current_time, duration)
                    
                    self.vehicle_point_data.append({
                        'id': veh.veh_id,
                        'type': 'veh',
                        'location': [veh.curr_lat, veh.curr_lon],
                        'duration': [veh.relocation_complete_time, self.current_time]
                    })
                    
                    veh.veh_state = 'not free'
                    veh.curr_grid = to_grid_id
                    veh.curr_lat = lat_d
                    veh.curr_lon = lon_d
                    veh.delay += duration
                    veh.relocation_routes = routes
                    veh.relocation_timestamps = timestamp
                    
                    self.curr_vehs[to_grid_id].append(veh)
    
    # def assign_veh_to_passenger(self) -> None:
    #     for grid_id, veh_lst in self.curr_vehs.items():
    #         filtered_veh_lst = list(filter(lambda x: x.veh_state == 'free' and x.delay == 0, veh_lst))
    #         filtered_passenger_lst = list(filter(lambda x: x.passenger_state == 'call', self.curr_passengers[grid_id]))
    #         if len(filtered_veh_lst) == 0 or len(filtered_passenger_lst) == 0: continue
            
    #         min_object_n = min(len(filtered_veh_lst), len(filtered_passenger_lst))
    #         random_vehs = np.random.choice(filtered_veh_lst, size=min_object_n, replace=False)
    #         random_passengers = np.random.choice(filtered_passenger_lst, size=min_object_n, replace=False)
            
    #         for veh, passenger in zip(random_vehs, random_passengers):
    #             lat1, lon1, lat2, lon2 = veh.curr_lat, veh.curr_lon, passenger.init_ori_lat, passenger.init_ori_lon
    #             routes, distance, duration = get_duration_distance((lon1, lat1, lon2, lat2), route=True)
    #             if len(routes) == 0:
    #                 raise
    #             timestamp = get_timestamp(routes, self.current_time, duration)
                
    #             veh.veh_state = 'assigned'
    #             veh.delay += duration
    #             veh.assigned_passenger_id = passenger.passenger_id
                
    #             passenger.passenger_state = 'wait'
    #             passenger.delay += duration
    #             passenger.assigned_vehicle_id = veh.veh_id
                
    #             self.vehicle_point_data.append({
    #                 'id': veh.veh_id,
    #                 'type': 'veh',
    #                 'location': [veh.curr_lat, veh.curr_lon],
    #                 'duration': [veh.relocation_complete_time, self.current_time]
    #             })
                
    #             self.passenger_point_data.append({
    #                 'id': passenger.passenger_id,
    #                 'type': 'passenger',
    #                 'location': [passenger.init_ori_lat, passenger.init_ori_lon],
    #                 'duration': [passenger.start_time, timestamp[-1]]
    #             })
                
    #             self.trip_data.append({
    #                 'id': veh.veh_id,r
    #                 'type': 'veh',
    #                 'state': 'assigned',
    #                 'route': list(map(lambda x: x[::-1], routes)),
    #                 'timestamp': timestamp
    #             })
    
    def assign_veh_to_passenger(self) -> None:
        vehicles = sum([veh_lst for veh_lst in self.curr_vehs.values()], [])
        free_vehicles = list(filter(lambda x: x.veh_state == 'free' and x.delay == 0, vehicles))
        relocating_vehicles = list(filter(lambda x: x.veh_state == 'not free', vehicles))
        vehicles = free_vehicles + relocating_vehicles
        
        passengers = sum([passenger_lst for passenger_lst in self.curr_passengers.values()], [])
        passengers = list(filter(lambda x: x.passenger_state == 'call', passengers))
        
        costs = []
        pairs = []
        for veh in vehicles:
            cost_lst = []
            pair_lst = []
            for passenger in passengers:
                cost = _haversine(veh.curr_lat, veh.curr_lon, passenger.init_ori_lat, passenger.init_ori_lon)
                cost_lst.append(cost)
                pair_lst.append({
                    'veh_grid': veh.curr_grid,
                    'veh_id': veh.veh_id,
                    'passenger_grid': passenger.init_ori_grid,
                    'passenger_id': passenger.passenger_id
                })
            costs.append(cost_lst)
            pairs.append(pair_lst)
        
        costs = np.array(costs)
        pairs = np.array(pairs)
        
        num_driver, num_passenger = costs.shape
        if num_driver < num_passenger:
            costs = costs.T
            num_driver, num_passenger = num_passenger, num_driver
            
        solver = pywraplp.Solver.CreateSolver('SCIP')
        
        if not solver: raise
        
        result_matrix = {}
        for i in range(num_driver):
            for j in range(num_passenger):
                result_matrix[i, j] = solver.IntVar(0, 1, '')
        
        for i in range(num_driver):
            solver.Add(solver.Sum([result_matrix[i, j] for j in range(num_passenger)]) <= 1)
        for j in range(num_passenger):
            solver.Add(solver.Sum([result_matrix[i, j] for i in range(num_driver)]) == 1)
        
        objective_terms = []
        for i in range(num_driver):
            for j in range(num_passenger):
                objective_terms.append(costs[i][j] * result_matrix[i, j])
        
        solver.Minimize(solver.Sum(objective_terms))
        status = solver.Solve()
        
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            for i in range(num_driver):
                for j in range(num_passenger):
                    if result_matrix[i, j].solution_value() > 0.5:
                        veh_grid = pairs[i, j]['veh_grid']
                        veh_id = pairs[i, j]['veh_id']
                        passenger_grid = pairs[i, j]['passenger_grid']
                        passenger_id = pairs[i, j]['passenger_id']

                        veh = list(filter(lambda x: x.veh_id == veh_id, self.curr_vehs[veh_grid]))[0]
                        passenger = list(filter(lambda x: x.passenger_id == passenger_id, self.curr_passengers[passenger_grid]))[0]
                        
                        if veh.veh_state == 'not free':
                            veh_last_indices = [idx for idx, t in enumerate(veh.relocation_timestamps) if t >= self.current_time]
                            veh_last_index = list(sorted(veh_last_indices))[0]
                            last_lat, last_lon = veh.relocation_routes[veh_last_index]
                            veh.curr_lat, veh.curr_lon = last_lat, last_lon
                            veh.delay = 0
                            veh.relocation_complete_time = veh.relocation_timestamps[veh_last_index]
                            
                            self.trip_data.append({
                                'id': veh.veh_id,
                                'type': 'veh',
                                'state': 'relocation',
                                'route': list(map(lambda x: x[::-1], veh.relocation_routes[:veh_last_index])),
                                'timestamp': veh.relocation_timestamps[:veh_last_index]
                            })
                        
                            veh.relocation_routes = []
                            veh.relocation_timestamps = []
                        
                        lat1, lon1, lat2, lon2 = veh.curr_lat, veh.curr_lon, passenger.init_ori_lat, passenger.init_ori_lon
                        routes, distance, duration = get_duration_distance((lon1, lat1, lon2, lat2), route=True)
                        if len(routes) == 0: raise
                        
                        timestamp = get_timestamp(routes, self.current_time, duration)
                        
                        veh.veh_state = 'assigned'
                        veh.delay += duration
                        veh.assigned_passenger_id = passenger.passenger_id
                        
                        passenger.passenger_state = 'wait'
                        passenger.delay += duration
                        passenger.assigned_vehicle_id = veh.veh_id
                
                        self.vehicle_point_data.append({
                            'id': veh.veh_id,
                            'type': 'veh',
                            'location': [veh.curr_lat, veh.curr_lon],
                            'duration': [veh.relocation_complete_time, self.current_time]
                        })
                        
                        self.passenger_point_data.append({
                            'id': passenger.passenger_id,
                            'type': 'passenger',
                            'location': [passenger.init_ori_lat, passenger.init_ori_lon],
                            'duration': [passenger.start_time, timestamp[-1]]
                        })
                        
                        self.trip_data.append({
                            'id': veh.veh_id,
                            'type': 'veh',
                            'state': 'assigned',
                            'route': list(map(lambda x: x[::-1], routes)),
                            'timestamp': timestamp
                        })
        else: raise
    
    def go_to_destination(self) -> None:
        for grid_id, veh_lst in self.curr_vehs.items():
            filtered_assigned_veh_lst = list(filter(lambda x: x.veh_state == 'assigned' and x.delay == 0, veh_lst))
            for veh in filtered_assigned_veh_lst:
                poped_veh_id = [idx for idx, veh_ in enumerate(self.curr_vehs[grid_id]) if veh_.veh_id == veh.veh_id][0]
                veh = self.curr_vehs[grid_id].pop(poped_veh_id)
                
                target_passenger = list(filter(lambda x: x.passenger_id == veh.assigned_passenger_id, self.total_passengers))[0]
                veh.curr_grid = target_passenger.init_ori_grid
                veh.curr_lat = target_passenger.init_ori_lat
                veh.curr_lon = target_passenger.init_ori_lon
                
                lat1, lon1, lat2, lon2 = veh.curr_lat, veh.curr_lon, target_passenger.init_dest_lat, target_passenger.init_dest_lon
                routes, distance, duration = get_duration_distance((lon1, lat1, lon2, lat2), route=True)
                if len(routes) == 0: raise
                timestamp = get_timestamp(routes, self.current_time, duration)
                
                veh.veh_state = 'move'
                veh.delay += duration
                
                target_passenger.passenger_state = 'move'
                target_passenger.delay += duration
                
                self.curr_vehs[veh.curr_grid].append(veh)
                
                self.trip_data.append({
                    'id': veh.veh_id,
                    'type': 'veh',
                    'state': 'in-service',
                    'route': list(map(lambda x: x[::-1], routes)),
                    'timestamp': timestamp
                })
            
            filtered_moved_veh_lst = list(filter(lambda x: x.veh_state == 'move' and x.delay == 0, veh_lst))
            for veh in filtered_moved_veh_lst:
                poped_veh_id = [idx for idx, veh_ in enumerate(self.curr_vehs[grid_id]) if veh_.veh_id == veh.veh_id][0]
                veh = self.curr_vehs[grid_id].pop(poped_veh_id)
                
                target_passenger = list(filter(lambda x: x.passenger_id == veh.assigned_passenger_id, self.total_passengers))[0]
                veh.curr_grid = target_passenger.init_dest_grid
                veh.curr_lat = target_passenger.init_dest_lat
                veh.curr_lon = target_passenger.init_dest_lon
                
                veh.veh_state = 'free'
                veh.assigned_passenger_id = None
                veh.relocation_complete_time = self.current_time
                target_passenger.passenger_state = 'arrive'
                
                self.curr_vehs[veh.curr_grid].append(veh)
    
    def _update_object(self) -> None:
        ### append veh in veh_pool
        curr_veh_id_lst = [veh.veh_id for veh_lst in self.curr_vehs.values() for veh in veh_lst]
        for veh in self.total_vehs:
            if veh.veh_id not in curr_veh_id_lst and veh.start_time <= self.current_time:
                self.curr_vehs[veh.init_grid].append(veh)
        
        ### update vehicles
        for veh_lst in self.curr_vehs.values():
            for veh in veh_lst:
                veh.delay = max(veh.delay - 1, 0)
                if veh.delay == 0 and veh.veh_state == 'not free':
                    veh.veh_state = 'free'
                    veh.relocation_complete_time = self.current_time
                    self.trip_data.append({
                        'id': veh.veh_id,
                        'type': 'veh',
                        'state': 'relocation',
                        'route': list(map(lambda x: x[::-1], veh.relocation_routes)),
                        'timestamp': veh.relocation_timestamps
                    })

        ### append passenger in passenger_pool
        curr_passenger_id_lst = [passenger.passenger_id for passenger_lst in self.curr_passengers.values() for passenger in passenger_lst]
        for passenger in self.total_passengers:
            if passenger.passenger_id not in curr_passenger_id_lst and passenger.start_time <= self.current_time:
                self.curr_passengers[passenger.init_ori_grid].append(passenger)
        
    def _update_after_running(self) -> None:
        predict_calling_passengers = {grid_id:0 for grid_id in self.grids.keys()}
        predict_arrival_passengers = {grid_id:0 for grid_id in self.grids.keys()}
        
        ### 예측값
        # 기존의 차량이 할당되지 않은 승객
        for grid_id, passenger_lst in self.curr_passengers.items():
            filtered_passenger_lst = list(filter(lambda x: x.passenger_state == 'call', passenger_lst))
            predict_calling_passengers[grid_id] += len(filtered_passenger_lst)
        
        # 재배치 될 시간 기준 생성될 예측 승객
        curr_passenger_id_lst = [passenger.passenger_id for passenger_lst in self.curr_passengers.values() for passenger in passenger_lst]
        for passenger in self.total_passengers:
            if passenger.passenger_id not in curr_passenger_id_lst and passenger.start_time <= self.current_time + 30:
                predict_calling_passengers[passenger.init_ori_grid] += 1
        
        # 도착 예측 승객 수
        total_veh_lst = sum([veh_lst for veh_lst in self.curr_vehs.values()], [])
        for grid_id, passenger_lst in self.curr_passengers.items():
            for passenger in passenger_lst:
                if passenger.passenger_state == 'wait':
                    veh = list(filter(lambda x: x.assigned_passenger_id == passenger.passenger_id, total_veh_lst))[0]
                    from_grid_id, to_grid_id = passenger.init_ori_grid, passenger.init_dest_grid
                    predicted_time = self.od_matrix[from_grid_id, to_grid_id]['time']
                    if veh.delay + predicted_time <= 30:
                        predict_arrival_passengers[passenger.init_dest_grid] += 1
                
                if passenger.passenger_state == 'move':
                    veh = list(filter(lambda x: x.assigned_passenger_id == passenger.passenger_id, total_veh_lst))[0]
                    if veh.delay <= 30:
                        predict_arrival_passengers[passenger.init_dest_grid] += 1
        
        self.predict_calling_passenger = predict_calling_passengers
        self.predict_arrival_passenger = predict_arrival_passengers
    
    def step(self) -> None:
        self.current_time += 1
        self._update_object()
    
    def _setup(self) -> None:
        if self.generate_data:
            generate_grid(self.place_name, self.unit, self.unit_size)
            generate_od_matrix()
            generate_random_data(self.place_name, self.n_taxi_point, 'drive')
            generate_random_data(self.place_name, self.n_passenger_point, 'walk')
        
        raw_grid = gpd.read_file('data/grid.json').to_dict('records')
        self.grids = {grid['index']:grid['geometry'] for grid in raw_grid}
        
        raw_od_matrix = pd.read_json('data/od_matrix.json').to_dict('records')
        self.od_matrix = {}
        for od in raw_od_matrix:
            self.od_matrix[od['index_o'], od['index_d']] = {
                'lat_o': od['lat_o'],
                'lon_o': od['lon_o'],
                'lat_d': od['lat_d'],
                'lon_d': od['lon_d'],
                'dist': od['distance'],
                'time': od['duration'],
                'random_points_o': od['random_points_o'],
                'random_points_d': od['random_points_d'],
            }
            
        if self.use_random_data:
            vehs = pd.read_json('data/random_vehicle.json').to_dict('records')
            passengers = pd.read_json('data/random_passenger.json').to_dict('records')
            
            self.total_vehs = []
            for veh in vehs:
                self.total_vehs.append(Vehicle(veh))
            
            self.total_passengers = []
            for passenger in passengers:
                self.total_passengers.append(Passenger(passenger))
        else:
            raise NotImplementedError()
    
    def _save_to_json(self) -> None:
        with open('result/trip.json', 'w') as f:
            json.dump(self.trip_data, f)
        
        with open('result/passenger.json', 'w') as f:
            json.dump(self.passenger_point_data, f)
        
        not_assigned_passenger_lst = []
        total_passenger = sum(list(map(lambda x: list(x), self.curr_passengers.values())), [])
        for x in list(filter(lambda x: x.passenger_state == 'call', total_passenger)):
            not_assigned_passenger_lst.append({
                'id': x.passenger_id,
                'type': 'passenger',
                'location': [x.init_ori_lat, x.init_ori_lon],
                'duration': [x.start_time, 1441]
            })
            
        with open('result/not_assigned_passenger.json', 'w') as f:
            json.dump(not_assigned_passenger_lst, f)
        
        with open('result/wait_vehs.json', 'w') as f:
            json.dump(self.vehicle_point_data, f)