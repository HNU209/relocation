from shapely.geometry import Point, Polygon
from shapely.prepared import prep
from pyproj import Transformer
import geopandas as gpd
import pandas as pd
import numpy as np
import osmnx as ox
import random

def _generate_grid(args):
    place_name = args['place_name']
    unit = args['unit']
    unit_size = args['unit_size']
    
    region_polygon_gdf = ox.geocode_to_gdf(place_name)
    region_info = region_polygon_gdf.iloc[0].to_dict()
    
    region_geometry = region_info['geometry']
    bbox = [region_info['bbox_west'], region_info['bbox_south'], region_info['bbox_east'], region_info['bbox_north']]
    
    transformer = Transformer.from_crs(4326, 5179, always_xy=True)
    bbox_meter = list(transformer.transform(bbox[0], bbox[1])) + list(transformer.transform(bbox[2], bbox[3]))
    
    if unit not in ['m', 'km']:
        NotImplementedError()
    else:
        if unit == 'km':
            unit_size *= 1000
        grids = _partition(region_geometry, bbox_meter, transformer, unit_size)
        grid_gdf = gpd.GeoDataFrame(grids, columns=['geometry']).reset_index()
        grid_gdf.to_file(f'data/grid.json', driver='GeoJSON')
        return grid_gdf

def _partition(geometry, bbox, transformer, unit_size):
    prepared_geometry = prep(geometry)
    
    grids = []
    for x in range(int(bbox[0]), int(bbox[2]), unit_size):
        for y in range(int(bbox[1]), int(bbox[3]), unit_size):
            corner_5179 = [(x, y), (x + unit_size, y), (x + unit_size, y + unit_size), (x, y + unit_size)]
            corner_4326 = [transformer.transform(c[0], c[1], direction='INVERSE') for c in corner_5179]
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

def _generate_od_matrix():
    grid_gdf = gpd.read_file('data/grid.json')
    grid_gdf['centroid'] = grid_gdf['geometry'].map(lambda x: list(x.centroid.coords)[0])
    grid_gdf['lat'] = grid_gdf['centroid'].map(lambda x: x[1])
    grid_gdf['lon'] = grid_gdf['centroid'].map(lambda x: x[0])
    grid_gdf = grid_gdf.drop(columns=['geometry'], axis=1)
    
    od_matrix_gdf = pd.merge(grid_gdf, grid_gdf, how='cross', suffixes=['_o', '_d'])
    
    distance = _haversine(od_matrix_gdf['lat_o'], od_matrix_gdf['lon_o'], od_matrix_gdf['lat_d'], od_matrix_gdf['lon_d'])
    od_matrix_gdf['distance'] = distance
    od_matrix_gdf['duration'] = 1 / (30 * (1 / distance)) * 60
    od_matrix_gdf.to_json('data/od_matrix.json', 'records')

def find_grid_id(point, grids):
    for grid_id, grid in grids.items():
        p = Point(point)
        if not grid.intersection(p).is_empty:
            return grid_id
    raise

def _generate_random_data(args, n_point, network_type, grids):
    place_name = args['place_name']
    
    if network_type not in ['drive', 'walk']:
        NotImplementedError()
    
    custom_filter = '["highway"~"primary|secondary|tertiary"]'
    graph = ox.graph_from_place(place_name, network_type=network_type, custom_filter=custom_filter)
    edges = ox.graph_to_gdfs(graph, nodes=False)
    edge_geometry = edges.reset_index()['geometry'].tolist()
    
    data = []
    data_type = 'vehicle' if network_type == 'drive' else 'passenger'
    for n in range(n_point):
        random_linestring = random.choice(edge_geometry)
        lon, lat = random.choice(list(random_linestring.coords))
        
        if data_type == 'vehicle':
            data.append({
                'id': n,
                'start_time': random.choice(range(1)),
                'grid_id': find_grid_id((lon, lat), grids),
                'lat': lat,
                'lon': lon
            })
        else:
            while 1:
                random_linestring_ = random.choice(edge_geometry)
                dest_lon, dest_lat = random.choice(list(random_linestring_.coords))
                if _haversine(lat, lon, dest_lat, dest_lon) > 5:
                    data.append({
                        'id': n,
                        'start_time': random.choice(range(0, 1400)),
                        'grid_id': find_grid_id((lon, lat), grids),
                        'lat': lat,
                        'lon': lon,
                        'dest_grid_id': find_grid_id((dest_lon, dest_lat), grids),
                        'dest_lat': dest_lat,
                        'dest_lon': dest_lon
                    })
                    break
    
    result_df = pd.DataFrame(data)
    result_df.to_json(f'data/random_{data_type}.json', 'records')
    
def _setup(args):
    ### 격자 데이터 생성
    grids = _generate_grid(args)
    
    ### o-d matrix 생성
    _generate_od_matrix()
    
    ### 랜덤 승객 & 챠량 생성
    if args['random_data'] and args['generate_data']:
        grids = {grid['index']: grid['geometry'] for grid in grids.to_dict('records')}
        _generate_random_data(args, args['n_taxi_point'], 'drive', grids)
        _generate_random_data(args, args['n_passenger_point'], 'walk', grids)