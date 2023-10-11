from typing import Union, NamedTuple
from itertools import accumulate
import numpy as np
import polyline
import requests

def compute_straight_distance(lat1, lon1, lat2, lon2):
    # haversine
    km_constant = 3959 * 1.609344
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a)) 
    km = km_constant * c
    return km

def get_duration_distance(point: Union[list, tuple], route=False) -> tuple:
    info = request_osrm(point)
    
    if route:
        if len(info) == 0:
            return [], 0, 0
        else:
            routes = polyline.decode(info['geometry'])
            distance = info['distance'] / 1000
            duration = np.ceil(info['duration'] / 60 * 100) / 100
            return routes, distance, duration
    else:
        if len(info) == 0:
            return 0, 0
        else:
            distance = info['distance'] / 1000
            duration = np.ceil(info['duration'] / 60 * 100) / 100
            return distance, duration

def get_timestamp(routes, start_time, duration):
    if len(routes) == 2:
        return [start_time, start_time + duration]
    else:
        per = np.array([compute_straight_distance(
            routes[i][0], routes[i][1],
            routes[i + 1][0], routes[i + 1][1]) + 1e-5 for i in range(len(routes) - 1)])
        per /= np.sum(per)
        timestamp = np.array(list(accumulate(per * duration))) + start_time
        timestamp = timestamp.tolist()
        timestamp.insert(0, start_time)
        return timestamp

def request_osrm(point: Union[list, tuple]) -> dict:
    url = f'http://localhost:5000/route/v1/driving/{point[0]},{point[1]};{point[2]},{point[3]}?overview=full'
    result_json = requests.get(url).json()
    if result_json['code'] == 'InvalidValue':
        return {}
    else:
        info = result_json['routes'][0]
        return info