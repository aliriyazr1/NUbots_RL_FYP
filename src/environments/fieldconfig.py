import os
from pathlib import Path
import yaml

class FieldConfig:
    """Class to handle field configuration loading and calculations"""
    
    def __init__(self, config_path="field_config.yaml"):
        # If config_path is just a filename, look for it relative to this script's directory
        if not os.path.isabs(config_path) and not os.path.dirname(config_path):
            script_dir = Path(__file__).parent
            self.config_path = script_dir / config_path
        else:
            self.config_path = config_path
                   
        self.config = self._load_config()
        self._calculate_derived_values()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                print(f"Warning: Config file {self.config_path} not found. Using default kidsize configuration.")
                return self._get_default_config()
            
            with open(config_file, 'r') as file:
                config = yaml.safe_load(file)
                print(f"Loaded configuration from {self.config_path}")
                return config
        except Exception as e:
            print(f"Error loading config: {e}. Using default configuration.")
            return self._get_default_config()
    
    def _get_default_config(self):
        """Return default kidsize configuration if file loading fails"""
        return {
            'field_type': 'kidsize',
            'real_world_dimensions': {
                'kidsize': {
                    'field_length': 9.0, 'field_width': 6.0, 'goal_depth': 0.6,
                    'goal_width': 2.6, 'goal_height': 1.2, 'goal_area_length': 1.0,
                    'goal_area_width': 3.0, 'penalty_mark_distance': 1.5,
                    'centre_circle_diameter': 1.5, 'border_strip_width': 1.0,
                    'penalty_area_length': 2.0, 'penalty_area_width': 5.0,
                    'line_width': 0.05
                }
            },
            'display_dimensions': {'width': 900, 'height': 600, 'scale_factor': 100},
            'robot_parameters': {
                'robot_radius': 0.15,
                'ball_radius': 0.08,
                'possession_threshold': 0.25,
                'base_robot_speed_mps': 2.5,  # metres per second
                'base_rotation_speed_rps': 0.67,  # rotations per second (1 full rotation per 1.5 sec)
                'base_opponent_speed_mps': 1.4  # opponent speed in metres per second
            },
            'physics': {'ball_mass': 0.5, 'ball_friction': 0.8, 'ball_bounce': 0.2, 
                       'ball_max_speed': 3.0, 'push_force_multiplier': 1.2},
            'difficulty_settings': {
                'easy': {'possession_distance': 0.5, 'collision_distance': 0.2, 'max_steps': 500, 'opponent_speed': 0.8},
                'medium': {'possession_distance': 0.4, 'collision_distance': 0.25, 'max_steps': 800, 'opponent_speed': 1.0},
                'hard': {'possession_distance': 0.35, 'collision_distance': 0.3, 'max_steps': 1000, 'opponent_speed': 1.2}
            },
            'reward_parameters': {
                'goal_scored': 15.0, 'time_penalty': -0.002, 'collision_penalty': -1.0,
                'edge_penalty': -1.0, 'corner_penalty': -0.5, 'center_field_bonus': 0.3
            }
        }
    
    def _calculate_derived_values(self):
        """Calculate derived values like pixel scale and strategic zones"""
        field_type = self.config['field_type']
        
        if field_type == 'custom':
            self.real_dims = self.config['custom_dimensions']
        else:
            self.real_dims = self.config['real_world_dimensions'][field_type]
        
        # Calculate pixel scale factors
        display_dims = self.config['display_dimensions']
        self.pixels_per_meter_x = display_dims['width'] / self.real_dims['field_length']
        self.pixels_per_meter_y = display_dims['height'] / self.real_dims['field_width']
        
        # Use consistent scale factor (smaller of the two to ensure field fits)
        self.pixels_per_meter_cached = min(self.pixels_per_meter_x, self.pixels_per_meter_y)
        
        # Calculate actual pixel dimensions
        self.field_width_pixels = int(self.real_dims['field_length'] * self.pixels_per_meter_cached)
        self.field_height_pixels = int(self.real_dims['field_width'] * self.pixels_per_meter_cached)
        
        # Goal dimensions in pixels
        self.goal_width_pixels = int(self.real_dims['goal_width'] * self.pixels_per_meter_cached)
        self.goal_depth_pixels = int(self.real_dims['goal_depth'] * self.pixels_per_meter_cached)
        
        # Calculate strategic zones in pixel coordinates
        self._calculate_strategic_zones()
    
    def _calculate_strategic_zones(self):
        """Calculate strategic zone coordinates in pixels"""
        zones = self.config.get('strategic_zones', {})
        self.strategic_zones = {}
        
        # Calculate optimal shooting zone
        if 'optimal_shooting' in zones:
            zone = zones['optimal_shooting']
            self.strategic_zones['optimal_shooting'] = {
                'x_min': int(zone['x_min_percent'] * self.field_width_pixels),
                'x_max': int(zone['x_max_percent'] * self.field_width_pixels),
                'y_min': int(zone['y_min_percent'] * self.field_height_pixels),
                'y_max': int(zone['y_max_percent'] * self.field_height_pixels)
            }
        
        # Calculate attacking zone
        if 'attacking_zone' in zones:
            zone = zones['attacking_zone']
            self.strategic_zones['attacking_zone'] = {
                'x_min': int(zone['x_min_percent'] * self.field_width_pixels),
                'x_max': int(zone['x_max_percent'] * self.field_width_pixels),
                'y_min': int(zone['y_min_percent'] * self.field_height_pixels),
                'y_max': int(zone['y_max_percent'] * self.field_height_pixels)
            }
        
        # Calculate goal area coordinates
        goal_center_y = self.field_height_pixels // 2
        goal_half_width = self.goal_width_pixels // 2
        self.strategic_zones['goal_area'] = {
            'x_min': self.field_width_pixels - int(self.real_dims['goal_area_length'] * self.pixels_per_meter),
            'x_max': self.field_width_pixels,
            'y_min': goal_center_y - int(self.real_dims['goal_area_width'] * self.pixels_per_meter // 2),
            'y_max': goal_center_y + int(self.real_dims['goal_area_width'] * self.pixels_per_meter // 2)
        }
        
        # Calculate penalty area coordinates
        self.strategic_zones['penalty_area'] = {
            'x_min': self.field_width_pixels - int(self.real_dims['penalty_area_length'] * self.pixels_per_meter),
            'x_max': self.field_width_pixels,
            'y_min': goal_center_y - int(self.real_dims['penalty_area_width'] * self.pixels_per_meter // 2),
            'y_max': goal_center_y + int(self.real_dims['penalty_area_width'] * self.pixels_per_meter // 2)
        }
    
    def meters_to_pixels(self, meters):
        """Convert meters to pixels"""
        if not hasattr(self, 'pixels_per_meter') or self.pixels_per_meter <= 0:
            raise ValueError("Invalid pixels_per_meter value. Check field configuration.")
        return (meters * self.pixels_per_meter)
    
    def meters_to_pixels_int(self, meters):
        """Convert meters to pixels as integer for pygame drawing"""
        return int(self.meters_to_pixels(meters))
    
    #TODO:  This is not used
    # def pixels_to_meters(self, pixels):
    #     """Convert pixels to meters"""
    #     if not hasattr(self, 'pixels_per_meter') or self.pixels_per_meter <= 0:
    #         raise ValueError("Invalid pixels_per_meter value. Check field configuration.")
    #     return pixels / self.pixels_per_meter
    @property
    def pixels_per_meter(self):
        """Calculate pixels per meter based on field dimensions"""
        if hasattr(self, '_pixels_per_meter_cached'):
            return self._pixels_per_meter_cached
    
        field_type = self.config['field_type']
        field_length_meters = self.config['real_world_dimensions'][field_type]['field_length']
        display_width = self.config['display_dimensions']['width']
        return display_width / field_length_meters
        
    # TODO: This is not used
    def is_in_zone(self, x, y, zone_name) -> bool:
        """Check if coordinates are within a strategic zone"""
        if zone_name not in self.strategic_zones:
            return False
        
        zone = self.strategic_zones[zone_name]
        return (zone['x_min'] <= x <= zone['x_max'] and 
                zone['y_min'] <= y <= zone['y_max'])
    
    def is_ball_in_goal(self, ball_x, ball_y) -> bool:
        """Check if ball is in goal"""
        goal_center_y = self.field_height_pixels // 2
        goal_half_width = self.goal_width_pixels // 2
        
        return (ball_x >= self.field_width_pixels and
                goal_center_y - goal_half_width <= ball_y <= goal_center_y + goal_half_width)
