from collections import defaultdict
from .utils import _setup
import geopandas as gpd
from .routing import *
from object import *
import pandas as pd
import random
import yaml
import json
import os

class Env:
    def __init__(self, args):
        self.args = args
        self.grids, self.od_matrix, self.vehs, self.passengers = self.setup()
        
        self.current_time = 0
        self.is_done = False
        
        ### save data
        self.vehicle_point_data = []
        self.passenger_point_data = []
        self.trip_data = []
        
        self.reset()
    
    def get_matrix_space(self):
        return (len(self.grids), len(self.grids))
    
    def reset(self):
        self.curr_vehs = {grid_id: [] for grid_id in self.grids.keys()}
        self.curr_passengers = {grid_id: [] for grid_id in self.grids.keys()}
        self.predict_calling_passengers = {grid_id: 0 for grid_id in self.grids.keys()}
        self.predict_arrival_passengers = {grid_id: 0 for grid_id in self.grids.keys()}
        
        total_veh = sum([veh_lst for veh_lst in self.curr_vehs.values()], [])
        
        ### initialize vehicle objects
        for veh in self.vehs:
            if veh.start_time <= self.current_time:
                self.curr_vehs[veh.init_grid].append(veh)
        
        ### initialize passenger objects
        for passenger in self.passengers:
            if passenger.start_time <= self.current_time:
                self.curr_passengers[passenger.init_ori_grid].append(passenger)
            
            ### predicted calling passenger
            if passenger.start_time <= self.current_time + 30:
                self.predict_calling_passengers[passenger.init_ori_grid] += 1
            
            ### predicted arrival passenger
            if passenger.passenger_state == 'wait':
                veh = list(filter(lambda x: x.assigned_passenger_id == passenger.passenger_id, total_veh))[0]
                from_grid_id, to_grid_id = passenger.init_ori_grid, passenger.init_dest_grid
                pred_time = self.od_matrix[from_grid_id, to_grid_id]['time']
                if veh.delay + pred_time <= 30:
                    self.predict_arrival_passengers[passenger.init_dest_grid] += 1
            
            if passenger.passenger_state == 'move':
                veh = list(filter(lambda x: x.assigned_passenger_id == passenger.passenger_id, total_veh))[0]
                if veh.delay <= 30:
                    self.predict_arrival_passengers[passenger.init_dest_grid] += 1
        
        self.current_time = 0
        self.is_done = False
    
    def get_state(self):
        ### 대기중인 차량
        sorted_veh_dict = dict(sorted(self.curr_vehs.items(), key=lambda x: x[0]))
        sorted_total_veh = [len(list(filter(lambda x: x.veh_state == 'free', veh_lst))) for veh_lst in sorted_veh_dict.values()]
        
        ### 예측 수요 기반 승객
        ### (2023-10-10 : 예측 모형 X)
        predict_calling_passengers_dict = dict(sorted(self.predict_calling_passengers.items(), key=lambda x: x[0]))
        predict_calling_passengers_lst = list(predict_calling_passengers_dict.values())
        
        predict_arrival_passengers_dict = dict(sorted(self.predict_arrival_passengers.items(), key=lambda x: x[0]))
        predict_arrival_passengers_lst = list(predict_arrival_passengers_dict.values())
        return sorted_total_veh, predict_calling_passengers_lst, predict_arrival_passengers_lst
    
    def step(self):
        self.current_time += 1
        self._update_object()
    
    def ga_step(self, population):
        for from_grid_id, row in enumerate(population.matrix):
            if sum(row) == 0: continue
            
            for to_grid_id, col in enumerate(row):
                ### 재배치 후 처리 필요
                if from_grid_id == to_grid_id or col == 0:
                    continue
                
                od_data = self.od_matrix[from_grid_id, to_grid_id]
                avail_veh_lst = list(filter(lambda x: x.veh_state == 'free', self.curr_vehs[from_grid_id]))
                random_veh_lst = np.random.choice(avail_veh_lst, size=int(col), replace=False)
                
                for idx, veh in enumerate(random_veh_lst):
                    poped_veh_id = [idx for idx, veh_ in enumerate(self.curr_vehs[from_grid_id]) if veh_.veh_id == veh.veh_id][0]
                    veh = self.curr_vehs[from_grid_id].pop(poped_veh_id)
                    
                    lat1, lon1, lat2, lon2 = veh.curr_lat, veh.curr_lon, od_data['lat_d'], od_data['lon_d']
                    routes, distance, duration = get_duration_distance((lon1, lat1, lon2, lat2), route=True)
                    timestamp = get_timestamp(routes, self.current_time, duration)
                    
                    self.vehicle_point_data.append({
                        'id': veh.veh_id,
                        'type': 'veh',
                        'location': [veh.curr_lat, veh.curr_lon],
                        'duration': [veh.relocation_complete_time, self.current_time]
                    })
                    
                    veh.veh_state = 'not free'
                    veh.curr_grid = to_grid_id
                    veh.curr_lat = od_data['lat_d']
                    veh.curr_lon = od_data['lon_d']
                    veh.delay += duration
                    
                    self.curr_vehs[to_grid_id].append(veh)
                    self.trip_data.append({
                        'id': veh.veh_id,
                        'type': 'veh',
                        'state': 'relocation',
                        'route': list(map(lambda x: x[::-1], routes)),
                        'timestamp': timestamp
                    })
    
    def assign_veh_to_passenger(self):
        for grid_id, veh_lst in self.curr_vehs.items():
            filtered_veh_lst = list(filter(lambda x: x.veh_state == 'free', veh_lst))
            filtered_passenger_lst = list(filter(lambda x: x.passenger_state == 'call', self.curr_passengers[grid_id]))
            
            if len(filtered_veh_lst) == 0 or len(filtered_passenger_lst) == 0: continue
            
            min_object_n = min(len(filtered_veh_lst), len(filtered_passenger_lst))
            random_vehs = np.random.choice(filtered_veh_lst, size=min_object_n, replace=False)
            random_passengers = np.random.choice(filtered_passenger_lst, size=min_object_n, replace=False)
            
            for veh, passenger in zip(random_vehs, random_passengers):
                lat1, lon1, lat2, lon2 = veh.curr_lat, veh.curr_lon, passenger.init_ori_lat, passenger.init_ori_lon
                routes, distance, duration = get_duration_distance((lon1, lat1, lon2, lat2), route=True)
                if len(routes) == 0:
                    raise
                timestamp = get_timestamp(routes, self.current_time, duration)
                
                veh.veh_state = 'assigned'
                veh.delay += duration
                veh.assigned_passenger_id = passenger.passenger_id
                
                passenger.passenger_state = 'wait'
                passenger.delay += duration
                
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
    
    def go_to_destination(self):
        for grid_id, veh_lst in self.curr_vehs.items():
            filtered_assigned_veh_lst = list(filter(lambda x: x.veh_state == 'assigned' and x.delay == 0, veh_lst))
            for veh in filtered_assigned_veh_lst:
                target_passenger = list(filter(lambda x: x.passenger_id == veh.assigned_passenger_id, self.passengers))[0]
                veh.curr_grid = target_passenger.init_ori_grid
                veh.curr_lat = target_passenger.init_ori_lat
                veh.curr_lon = target_passenger.init_ori_lon
                
                lat1, lon1, lat2, lon2 = veh.curr_lat, veh.curr_lon, target_passenger.init_dest_lat, target_passenger.init_dest_lon
                routes, distance, duration = get_duration_distance((lon1, lat1, lon2, lat2), route=True)
                timestamp = get_timestamp(routes, self.current_time, duration)
                
                veh.veh_state = 'move'
                veh.delay += duration
                
                target_passenger.passenger_state = 'move'
                target_passenger.delay += duration
                
                self.trip_data.append({
                    'id': veh.veh_id,
                    'type': 'veh',
                    'state': 'in-service',
                    'route': list(map(lambda x: x[::-1], routes)),
                    'timestamp': timestamp
                })
            
            filtered_moved_veh_lst = list(filter(lambda x: x.veh_state == 'move' and x.delay == 0, veh_lst))
            for veh in filtered_moved_veh_lst:
                target_passenger = list(filter(lambda x: x.passenger_id == veh.assigned_passenger_id, self.passengers))[0]
                veh.curr_grid = target_passenger.init_dest_grid
                veh.curr_lat = target_passenger.init_dest_lat
                veh.curr_lon = target_passenger.init_dest_lon
                
                veh.veh_state = 'free'
                veh.assigned_passenger_id = None
                veh.relocation_complete_time = self.current_time
                target_passenger.passenger_state = 'arrive'

    def _update_object(self):
        ### update vehicles
        for veh_lst in self.curr_vehs.values():
            for veh in veh_lst:
                veh.delay = max(veh.delay - 1, 0)
                if veh.delay == 0 and veh.veh_state == 'not free':
                    veh.relocation_complete_time = self.current_time
                    veh.veh_state = 'free'
        
        ### update passengers
        total_curr_passenger_id_lst = [passenger.passenger_id for passenger_lst in self.curr_passengers.values() for passenger in passenger_lst]
        for passenger in self.passengers:
            passenger_id = passenger.passenger_id
            if passenger.start_time <= self.current_time:
                grid_id = passenger.init_ori_grid
                if passenger_id not in total_curr_passenger_id_lst:
                    self.curr_passengers[grid_id].append(passenger)
        
        predict_calling_passengers = {grid_id: 0 for grid_id in self.grids.keys()}
        predict_arrival_passengers = {grid_id: 0 for grid_id in self.grids.keys()}
        
        total_veh = sum([veh_lst for veh_lst in self.curr_vehs.values()], [])
        ### 발생 예측 승객 수
        # 기존의 승객 중 배차 안 된 승객
        for grid_id, passenger_lst in self.curr_passengers.items():
            filtered_passenger_lst = list(filter(lambda x: x.passenger_state == 'call', passenger_lst))
            predict_calling_passengers[grid_id] += len(filtered_passenger_lst)
        
        # 새로 차량 호출한 승객
        total_curr_passenger_id_lst = [passenger.passenger_id for passenger_lst in self.curr_passengers.values() for passenger in passenger_lst]
        for passenger in self.passengers:
            if passenger.passenger_id in total_curr_passenger_id_lst: continue
            if passenger.start_time <= self.current_time + 30:
                grid_id = passenger.init_ori_grid
                predict_calling_passengers[grid_id] += 1
        
        ### 도착 예측 승객 수
        for grid_id, passenger_lst in self.curr_passengers.items():
            for passenger in passenger_lst:
                ### predicted arrival passenger
                if passenger.passenger_state == 'wait':
                    veh = list(filter(lambda x: x.assigned_passenger_id == passenger.passenger_id, total_veh))[0]
                    from_grid_id, to_grid_id = passenger.init_ori_grid, passenger.init_dest_grid
                    pred_time = self.od_matrix[from_grid_id, to_grid_id]['time']
                    if veh.delay + pred_time <= 30:
                        predict_arrival_passengers[passenger.init_dest_grid] += 1
                
                if passenger.passenger_state == 'move':
                    veh = list(filter(lambda x: x.assigned_passenger_id == passenger.passenger_id, total_veh))[0]
                    if veh.delay <= 30:
                        predict_arrival_passengers[passenger.init_dest_grid] += 1
        
        self.predict_calling_passengers = predict_calling_passengers
        self.predict_arrival_passengers = predict_arrival_passengers
    
    def setup(self):
        ### download data
        _setup(self.args)
        
        ### setting data
        # load grid
        grid_lst = gpd.read_file('data/grid.json').to_dict('records')
        grid_dict = {grid['index'] : grid['geometry'] for grid in grid_lst}
        
        # load o-d matrix
        od_matrix_lst = pd.read_json('data/od_matrix.json').to_dict('records')
        od_matrix_dict = {}
        
        for od in od_matrix_lst:
            from_grid_id = od['index_o']
            to_grid_id = od['index_d']
            lat_o = od['lat_o']
            lon_o = od['lon_o']
            lat_d = od['lat_d']
            lon_d = od['lon_d']
            dist = od['distance']
            time = od['duration']
            
            od_matrix_dict[from_grid_id, to_grid_id] = {
                'lat_o': lat_o,
                'lon_o': lon_o,
                'lat_d': lat_d,
                'lon_d': lon_d,
                'dist': dist,
                'time': time
            }
        
        # load_object
        veh_objects, passenger_objects = [], []
        if self.args['random_data']:
            with open('data/random_vehicle.json', 'r') as f:
                vehs = json.load(f)
                for veh in vehs:
                    veh_objects.append(Vehicle(veh))
        
            with open('data/random_passenger.json', 'r') as f:
                passengers = json.load(f)
                for passenger in passengers:
                    passenger_objects.append(Passenger(passenger))
        else:
            NotImplementedError()
        return grid_dict, od_matrix_dict, veh_objects, passenger_objects
    
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