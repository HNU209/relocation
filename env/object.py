class Vehicle:
    def __init__(self, data: dict) -> None:
        self.data = data
        self.veh_id = self.data['id']
        self.start_time = self.data['start_time']
        
        self.init_grid = self.data['grid_id']
        self.init_lat = self.data['lat']
        self.init_lon = self.data['lon']
        
        '''
        veh_state:
            free: 'available relocation'
            not free: 'move to relocated location'
            assigned: 'move to passenger'
            move: 'move to destination'
        '''
        self.reset()
    
    def reset(self) -> None:
        self.veh_state = 'free'
        self.delay = 0
        
        self.curr_grid = self.init_grid
        self.curr_lat = self.init_lat
        self.curr_lon = self.init_lon
        
        self.assigned_passenger_id = None
        self.relocation_complete_time = 0
        
        self.relocation_routes = []
        self.relocation_timestamps = []

class Passenger:
    def __init__(self, data: dict) -> None:
        self.data = data
        self.passenger_id = self.data['id']
        self.start_time = self.data['start_time']
        
        self.init_ori_grid = self.data['grid_id']
        self.init_ori_lat = self.data['lat']
        self.init_ori_lon = self.data['lon']
        
        self.init_dest_grid = self.data['dest_grid_id']
        self.init_dest_lat = self.data['dest_lat']
        self.init_dest_lon = self.data['dest_lon']
        
        '''
        passenger_state:
            call: 'waiting for assignment to passenger'
            wait: 'waiting for vehicle to come'
            move: 'go to destination'
            arrive: 'arrive at destination location'
        '''
        self.reset()
    
    def reset(self) -> None:
        self.passenger_state = 'call'
        self.delay = 0
        
        self.curr_grid = self.init_ori_grid
        self.curr_lat = self.init_ori_lat
        self.curr_lon = self.init_ori_lon
        
        self.assigned_vehicle_id = None