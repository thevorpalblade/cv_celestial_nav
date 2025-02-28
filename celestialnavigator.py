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
        """
        Calculate latitude and longitude using celestial navigation.
        
        Args:
            sun_elevation: Measured elevation angle in degrees
            sun_bearing: Measured bearing (azimuth) in degrees
            timestamp: UTC timestamp of observation
            height_meters: Observer's height above sea level in meters
        
        Returns:
            tuple: (latitude, longitude) in degrees
        """
        # Apply corrections to elevation
        corrected_elevation = self._correct_for_dip(sun_elevation, height_meters)
        corrected_elevation = self._correct_for_refraction(corrected_elevation)
        corrected_elevation = self._correct_for_parallax(corrected_elevation, timestamp)
        
        # Convert elevation and bearing to radians
        h = self._degrees_to_radians(corrected_elevation)  # Height/altitude
        Az = self._degrees_to_radians(sun_bearing)         # Azimuth
        
        # Get sun's declination for the given time
        observer = ephem.Observer()
        observer.date = timestamp
        self.sun.compute(observer)
        d = float(self.sun.dec)  # Declination in radians
        
        # Calculate latitude using the navigational triangle
        # sin(h) = sin(d)sin(L) + cos(d)cos(L)cos(t)
        # where h is height, d is declination, L is latitude, t is hour angle
        
        # Calculate latitude using the altitude formula
        sin_L = (math.sin(h) - math.sin(d) * math.cos(Az)) / (math.cos(d) * math.sin(Az))
        L = math.asin(max(-1.0, min(1.0, sin_L)))  # Latitude in radians
        
        # Calculate Local Hour Angle (t)
        cos_t = (math.sin(h) - math.sin(d) * math.sin(L)) / (math.cos(d) * math.cos(L))
        t = math.acos(max(-1.0, min(1.0, cos_t)))
        
        # Adjust hour angle based on azimuth quadrant
        if 0 <= sun_bearing <= 180:
            t = -t  # Morning (sun in eastern sky)
        
        # Get Greenwich Mean Sidereal Time
        gmst = float(observer.sidereal_time())  # Returns GMST in hours
        gmst_rad = gmst * 2 * math.pi / 24.0    # Convert hours to radians
        
        # Calculate longitude
        # Longitude = Hour Angle + GMST
        lon = t + gmst_rad
        
        # Normalize longitude to [-π, π]
        lon = ((lon + math.pi) % (2 * math.pi)) - math.pi
        
        # Convert to degrees
        latitude = self._radians_to_degrees(L)
        longitude = self._radians_to_degrees(lon)
        
        return latitude, longitude

    def _correct_for_refraction(self, apparent_elevation):
        """Correct for atmospheric refraction."""
        # Skip correction for invalid elevations
        if apparent_elevation > 90 or apparent_elevation < -90:
            return apparent_elevation
        
        # Convert to radians
        h = self._degrees_to_radians(apparent_elevation)
        
        # Use more accurate refraction formula
        if apparent_elevation >= 15:
            # Saemundsson's formula for higher elevations
            R = 1.02 / math.tan(h + 10.3/(apparent_elevation + 5.11))
        else:
            # Bennet's formula for low elevations
            R = 1.0 / math.tan(h) + 0.0019279
        
        # Convert from arcminutes to degrees and subtract from apparent elevation
        return apparent_elevation - R/60.0

    def _correct_for_parallax(self, elevation, timestamp):
        """Correct for solar parallax."""
        observer = ephem.Observer()
        observer.date = timestamp
        self.sun.compute(observer)
        
        # Get sun's distance in AU
        sun_distance = self.sun.earth_distance
        
        # Earth radius in AU
        earth_radius = 6378137.0 / 149597870700.0
        
        # Calculate parallax correction
        parallax = math.asin(earth_radius / sun_distance)
        
        # Apply correction based on elevation
        h = self._degrees_to_radians(elevation)
        correction = self._radians_to_degrees(parallax * math.cos(h))
        
        return elevation + correction

    def _correct_for_dip(self, elevation, height_meters):
        """Correct for geometric dip of horizon."""
        if height_meters <= 0:
            return elevation
        
        # Calculate dip correction
        # More accurate formula including refraction
        dip = 0.0293 * math.sqrt(height_meters)
        
        return elevation - dip

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
    # Ensure timestamp is in UTC
    if timestamp.tzinfo is None:
        raise ValueError("Timestamp must include timezone information")
    timestamp = timestamp.astimezone(datetime.timezone.utc)
    
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