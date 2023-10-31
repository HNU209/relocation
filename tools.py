from shapely.geometry import Point, Polygon, MultiLineString
from typing import Union, Literal, Tuple, Callable
from shapely.prepared import prep
from itertools import accumulate
from pyproj import Transformer
import geopandas as gpd
import pandas as pd
import numpy as np
import osmnx as ox
import requests
import polyline
import random
import os

os.chdir('/home/happy956/relocation')

def generate_grid(place_name: str, unit: str, unit_size: int) -> None:
    region_polygon_gdf = ox.geocode_to_gdf(place_name)
    region_info = region_polygon_gdf.iloc[0].to_dict()
    region_geometry = region_info['geometry']
    bbox = [region_info['bbox_west'], region_info['bbox_south'], region_info['bbox_east'], region_info['bbox_north']]
    
    transformer = Transformer.from_crs(4326, 5179, always_xy=True).transform
    bbox_meter = list(transformer(bbox[0], bbox[1])) + list(transformer(bbox[2], bbox[3]))
    
    if unit not in ['m', 'km']:
        raise NotImplementedError()
    else:
        if unit == 'km':
            unit_size *= 1000
        grids = _partition(region_geometry, bbox_meter, transformer, unit_size)
        grid_gdf = gpd.GeoDataFrame(grids, columns=['geometry'])
        
        custom_filter = '["highway"~"primary|secondary|tertiary"]'
        graph = ox.graph_from_place('대전, 대한민국', network_type='drive', custom_filter=custom_filter)
        edges = ox.graph_to_gdfs(graph, nodes=False)
        edge_geometry = edges.reset_index()['geometry'].tolist()
        
        edges = MultiLineString(edge_geometry)
        grids = grid_gdf['geometry']
        intersectios_grids = grids.map(lambda x: x.intersection(edges))
        grid_gdf = grid_gdf[~intersectios_grids.map(lambda x: x.is_empty)].reset_index(drop=True).reset_index()
        grid_gdf.to_file('data/grid.json')

def generate_od_matrix() -> None:
    grid_gdf = gpd.read_file('data/grid.json')
    grid_gdf['centroid'] = grid_gdf['geometry'].map(lambda x: list(x.centroid.coords)[0])
    grid_gdf['lat'] = grid_gdf['centroid'].map(lambda x: x[1])
    grid_gdf['lon'] = grid_gdf['centroid'].map(lambda x: x[0])
    grid_gdf = grid_gdf.drop(columns=['geometry', 'centroid'], axis=1)
    
    od_matrix_gdf = pd.merge(grid_gdf, grid_gdf, how='cross', suffixes=['_o', '_d'])
    distance = _haversine(od_matrix_gdf['lat_o'], od_matrix_gdf['lon_o'], od_matrix_gdf['lat_d'], od_matrix_gdf['lon_d'])
    od_matrix_gdf['distance'] = distance
    od_matrix_gdf['duration'] = 1 / (30 * (1 / distance)) * 60
    od_matrix_gdf.to_json('data/od_matrix.json', 'records')

def generate_random_data(place_name: str, n_point: int, network_type: str) -> None:
    if network_type not in ['drive', 'walk']:
        raise NotImplementedError()
    
    custom_filter = '["highway"~"primary|secondary|tertiary"]'
    graph = ox.graph_from_place(place_name, network_type=network_type, custom_filter=custom_filter)
    edges = ox.graph_to_gdfs(graph, nodes=False)
    edge_geometry = edges.reset_index()['geometry'].tolist()
    
    data = []
    grids = gpd.read_file('data/grid.json')
    grids = {grid['index']:grid['geometry'] for grid in grids.to_dict('records')}
    data_type = 'vehicle' if network_type == 'drive' else 'passenger'
    
    if data_type == 'vehicle':
        for n in range(n_point):
            random_linestring = random.choice(edge_geometry)
            lon, lat = random.choice(list(random_linestring.coords))
            data.append({
                'id': n,
                'start_time': random.choice(range(1)),
                'grid_id': _find_grid_id((lon, lat), grids),
                'lat': lat,
                'lon': lon
            })
    
    elif data_type == 'passenger':
        for n in range(n_point):
            random_linestring = random.choice(edge_geometry)
            lon, lat = random.choice(list(random_linestring.coords))
            
            while 1:
                random_linestring_ = random.choice(edge_geometry)
                dest_lon, dest_lat = random.choice(list(random_linestring_.coords))
                if _haversine(lat, lon, dest_lat, dest_lon) > 5:
                    data.append({
                        'id': n,
                        'start_time': random.choice(range(0, 1400)),
                        'grid_id': _find_grid_id((lon, lat), grids),
                        'lat': lat,
                        'lon': lon,
                        'dest_grid_id': _find_grid_id((dest_lon, dest_lat), grids),
                        'dest_lat': dest_lat,
                        'dest_lon': dest_lon
                    })
                    break
    result_df = pd.DataFrame(data)
    result_df.to_json(f'data/random_{data_type}.json', 'records')

def get_duration_distance(point: Union[list, tuple], route: bool = False) -> Union[Tuple[list, Literal[0], Literal[0]], Tuple[list, float, float], Tuple[Literal[0], Literal[0]], Tuple[float, float]]:
    info = _request_osrm(point)
    
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

def get_timestamp(routes: list, start_time: Union[int, float], duration: float) -> np.ndarray:
    if len(routes) == 2:
        return [start_time, start_time + duration]
    else:
        per = np.array([_haversine(
            routes[i][0], routes[i][1],
            routes[i + 1][0], routes[i + 1][1]) + 1e-5 for i in range(len(routes) - 1)])
        per /= np.sum(per)
        timestamp = np.array(list(accumulate(per * duration))) + start_time
        timestamp = timestamp.tolist()
        timestamp.insert(0, start_time)
        return timestamp

def _partition(geometry: pd.Series, bbox: Union[list, tuple], transformer: Callable, unit_size: int) -> list:
    prepared_geometry = prep(geometry)
    
    grids = []
    for x in range(int(bbox[0]), int(bbox[2]), unit_size):
        for y in range(int(bbox[1]), int(bbox[3]), unit_size):
            corner_5179 = [(x, y), (x + unit_size, y), (x + unit_size, y + unit_size), (x, y + unit_size)]
            corner_4326 = [transformer(c[0], c[1], direction='INVERSE') for c in corner_5179]
            grid = Polygon(corner_4326)
            grids.append(grid)
    
    grids = list(filter(prepared_geometry.intersects, grids))
    return grids

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    km_constant = 6371
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distance_km = km_constant * c
    return distance_km

def _find_grid_id(point: Union[list, tuple], grids: dict) -> int:
    for grid_id, grid in grids.items():
        p = Point(point)
        if not grid.intersection(p).is_empty:
            return grid_id
    raise

def _request_osrm(point: Union[list, tuple]) -> dict:
    try:
        url = f'http://localhost:5000/route/v1/driving/{point[0]},{point[1]};{point[2]},{point[3]}?overview=full'
        result_json = requests.get(url).json()
        info = result_json['routes'][0]
    except:
        print(point)
        raise
    return info