# This is a program to use a webcam image to do celestial navigation.

# Import necessary libraries
import cv2
import numpy as np

import ephem
from datetime import datetime
import math
import numpy as np

class CelestialNavigator:
    def __init__(self):
        self.sun = ephem.Sun()
    
    def _degrees_to_radians(self, degrees):
        """Convert degrees to radians."""
        return degrees * math.pi / 180.0
    
    def _radians_to_degrees(self, radians):
        """Convert radians to degrees."""
        return radians * 180.0 / math.pi

    def calculate_position(self, sun_elevation, sun_bearing, timestamp, height_meters=0):
        """Calculate latitude and longitude with improved accuracy."""
        # Apply corrections
        corrected_elevation = self._correct_for_dip(sun_elevation, height_meters)
        corrected_elevation = self._correct_for_refraction(corrected_elevation)
        corrected_elevation = self._correct_for_parallax(corrected_elevation, timestamp)
        
        # Create observer to get sun's declination
        observer = ephem.Observer()
        observer.date = timestamp
        self.sun.compute(observer)
        
        # Get sun's declination
        declination = float(self.sun.dec)  # Already in radians
        
        # Convert our measurements to radians
        elevation = self._degrees_to_radians(corrected_elevation)
        azimuth = self._degrees_to_radians(sun_bearing)
        
        # Calculate hour angle (in radians)
        hour_angle = math.atan2(
            -math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation) * math.cos(declination) - 
            math.cos(elevation) * math.sin(declination) * math.cos(azimuth)
        )
        
        # Calculate latitude
        latitude = math.asin(
            math.sin(elevation) * math.sin(declination) +
            math.cos(elevation) * math.cos(declination) * math.cos(azimuth)
        )
        
        # Calculate longitude
        # Get Greenwich Mean Sidereal Time in radians
        gmst = ephem.gmst(observer.date) * 2 * math.pi / 24
        longitude = gmst - hour_angle
        
        # Normalize longitude to [-π, π]
        longitude = ((longitude + math.pi) % (2 * math.pi)) - math.pi
        
        return self._radians_to_degrees(latitude), self._radians_to_degrees(longitude)

    def _correct_for_refraction(self, apparent_elevation):
        """
        Correct for atmospheric refraction.
        
        Args:
            apparent_elevation: Measured elevation in degrees
        
        Returns:
            float: Corrected elevation in degrees
        """
        # Standard temperature (15°C) and pressure (1013.25 hPa)
        if apparent_elevation < 15:
            # More accurate formula for low elevations
            tan_e = math.tan(self._degrees_to_radians(apparent_elevation))
            refraction_correction = 1.02 / tan_e - 0.0019279
        else:
            # Simpler formula for higher elevations
            refraction_correction = 1.0 / math.tan(self._degrees_to_radians(apparent_elevation + 7.31)) - 0.0068

        return apparent_elevation - refraction_correction / 60  # Convert arcminutes to degrees

    def _correct_for_parallax(self, elevation, timestamp):
        """
        Correct for solar parallax.
        
        Args:
            elevation: Measured elevation in degrees
            timestamp: Time of observation
        
        Returns:
            float: Corrected elevation in degrees
        """
        observer = ephem.Observer()
        observer.date = timestamp
        self.sun.compute(observer)
        
        # Get sun's distance in AU
        sun_distance = self.sun.earth_distance
        
        # Earth radius in AU (approximate)
        earth_radius = 6378137.0 / 149597870700.0
        
        parallax = math.asin(earth_radius / sun_distance)
        parallax_correction = parallax * math.cos(self._degrees_to_radians(elevation))
        
        return elevation + self._radians_to_degrees(parallax_correction)

    def _correct_for_dip(self, elevation, height_meters):
        """
        Correct for geometric dip of horizon due to height above sea level.
        
        Args:
            elevation: Measured elevation in degrees
            height_meters: Height above sea level in meters
        
        Returns:
            float: Corrected elevation in degrees
        """
        # Pure geometric dip correction in degrees
        # Formula: dip = arccos(R/(R+h)) where R is Earth's radius
        earth_radius = 6371000  # meters
        dip_correction = math.acos(earth_radius / (earth_radius + height_meters))
        return elevation - self._radians_to_degrees(dip_correction)

def get_position(elevation, bearing, timestamp, height_meters=0):
    """
    Convenience function to get position from a single observation.
    
    Args:
        elevation (float): Sun elevation in degrees
        bearing (float): Compass bearing to sun in degrees
        timestamp (datetime): UTC timestamp of observation
        height_meters (float): Height above sea level in meters
        
    Returns:
        tuple: (latitude, longitude) in degrees
    """
    navigator = CelestialNavigator()
    return navigator.calculate_position(elevation, bearing, timestamp, height_meters)

class VesselTracker:
    def __init__(self, window_size=10):
        """
        Initialize vessel tracker.
        
        Args:
            window_size (int): Number of recent positions to keep for smoothing
        """
        self.positions = []  # List of (timestamp, lat, lon, elevation, bearing, height)
        self.window_size = window_size
        self.navigator = CelestialNavigator()
    
    def add_observation(self, elevation, bearing, timestamp, height_meters=0):
        """
        Add a new observation and update vessel position.
        
        Args:
            elevation (float): Sun elevation in degrees
            bearing (float): Compass bearing to sun in degrees
            timestamp (datetime): UTC timestamp of observation
            height_meters (float): Height above sea level in meters
        
        Returns:
            tuple: (smoothed_lat, smoothed_lon)
        """
        lat, lon = self.navigator.calculate_position(elevation, bearing, timestamp, height_meters)
        self.positions.append({
            'timestamp': timestamp,
            'latitude': lat,
            'longitude': lon,
            'elevation': elevation,
            'bearing': bearing,
            'height': height_meters
        })
        
        # Keep only recent positions
        if len(self.positions) > self.window_size:
            self.positions.pop(0)
        
        return self.get_smoothed_position()
    
    def get_smoothed_position(self):
        """
        Calculate smoothed position using weighted average of recent positions.
        More recent positions and those with higher sun elevations get higher weights.
        
        Returns:
            tuple: (latitude, longitude) or None if no positions
        """
        if not self.positions:
            return None
        
        weights = []
        latest_time = self.positions[-1]['timestamp']
        
        for pos in self.positions:
            # Time weight: exponential decay with time difference
            time_diff = (latest_time - pos['timestamp']).total_seconds()
            time_weight = math.exp(-time_diff / 3600)  # 1-hour time constant
            
            # Elevation weight: higher elevation = better accuracy
            elevation_weight = math.sin(self.navigator._degrees_to_radians(pos['elevation']))
            
            # Combined weight
            weights.append(time_weight * elevation_weight)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Calculate weighted average
        smoothed_lat = sum(pos['latitude'] * w for pos, w in zip(self.positions, weights))
        smoothed_lon = sum(pos['longitude'] * w for pos, w in zip(self.positions, weights))
        
        return smoothed_lat, smoothed_lon
    
    def get_speed_and_heading(self):
        """
        Calculate vessel's speed and heading from recent positions.
        
        Returns:
            tuple: (speed_knots, heading_degrees) or None if insufficient data
        """
        if len(self.positions) < 2:
            return None
        
        # Get two most recent positions
        pos1 = self.positions[-2]
        pos2 = self.positions[-1]
        
        # Calculate time difference in hours
        time_diff = (pos2['timestamp'] - pos1['timestamp']).total_seconds() / 3600
        
        # Calculate distance in nautical miles
        lat1, lon1 = pos1['latitude'], pos1['longitude']
        lat2, lon2 = pos2['latitude'], pos2['longitude']
        
        R = 3440.065  # Earth radius in nautical miles
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        lat1, lat2 = math.radians(lat1), math.radians(lat2)
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        distance = 2 * R * math.asin(math.sqrt(a))
        
        # Calculate speed in knots
        speed = distance / time_diff if time_diff > 0 else 0
        
        # Calculate heading
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        heading = math.degrees(math.atan2(y, x)) % 360
        
        return speed, heading
    
    def get_position_history(self):
        """
        Get full position history.
        
        Returns:
            list: List of position dictionaries
        """
        return self.positions.copy()

# Example usage:
# tracker = VesselTracker()
# lat, lon = tracker.add_observation(elevation=45.0, bearing=180.0, timestamp=datetime.utcnow())
# speed, heading = tracker.get_speed_and_heading()