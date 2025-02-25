import cv2
import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass
import math

@dataclass
class CameraLens:
    """Camera lens characteristics and distortion parameters."""
    # Basic parameters
    horizontal_fov: float  # Horizontal field of view in degrees
    vertical_fov: float   # Vertical field of view in degrees
    
    # Distortion parameters
    k1: float = 0.0  # Radial distortion coefficient 1
    k2: float = 0.0  # Radial distortion coefficient 2
    p1: float = 0.0  # Tangential distortion coefficient 1
    p2: float = 0.0  # Tangential distortion coefficient 2
    
    def __post_init__(self):
        """Calculate derived parameters."""
        self.aspect_ratio = self.horizontal_fov / self.vertical_fov

class SunDetector:
    # Angular diameter of the sun in degrees
    SUN_ANGULAR_DIAMETER = 0.5

    def __init__(self, 
                 lens: CameraLens,
                 camera_bearing: float = 0.0,  # Bearing of camera center in degrees
                 min_sun_radius_factor: float = 0.8,  # Factor to account for detection variations
                 max_sun_radius_factor: float = 1.5   # Factor to account for blooming/glare
                ):
        self.lens = lens
        self.camera_bearing = camera_bearing
        
        # Calculate sun radius in pixels based on FOV
        # Use the smaller FOV (typically vertical) to ensure conservative estimates
        min_fov = min(self.lens.horizontal_fov, self.lens.vertical_fov)
        pixels_per_degree = 1000 / min_fov  # Assuming 1000px for standardization
        base_sun_radius = (self.SUN_ANGULAR_DIAMETER / 2) * pixels_per_degree
        
        # Apply factors to account for detection variations and blooming
        self.min_sun_radius = int(base_sun_radius * min_sun_radius_factor)
        self.max_sun_radius = int(base_sun_radius * max_sun_radius_factor)
        
    def _undistort_coordinates(self, x: float, y: float, 
                             image_width: int, image_height: int) -> Tuple[float, float]:
        """
        Convert distorted image coordinates to undistorted normalized coordinates.
        
        Args:
            x, y: Pixel coordinates
            image_width, image_height: Image dimensions
            
        Returns:
            tuple: (x, y) in normalized coordinates (-1 to 1)
        """
        # Convert to normalized coordinates (-1 to 1)
        x_norm = (x - image_width/2) / (image_width/2)
        y_norm = (y - image_height/2) / (image_height/2)
        
        # Convert to camera coordinates
        x_cam = x_norm
        y_cam = y_norm * self.lens.aspect_ratio
        
        # Calculate radius from center
        r2 = x_cam*x_cam + y_cam*y_cam
        
        # Apply radial distortion correction
        distortion_factor = 1 + self.lens.k1*r2 + self.lens.k2*r2*r2
        
        x_undist = x_cam * distortion_factor
        y_undist = y_cam * distortion_factor
        
        # Apply tangential distortion correction
        x_undist += (2*self.lens.p1*x_cam*y_cam + self.lens.p2*(r2 + 2*x_cam*x_cam))
        y_undist += (self.lens.p1*(r2 + 2*y_cam*y_cam) + 2*self.lens.p2*x_cam*y_cam)
        
        return x_undist, y_undist/self.lens.aspect_ratio

    def _pixel_to_angles(self, x: float, y: float, 
                        image_width: int, image_height: int) -> Tuple[float, float]:
        """
        Convert pixel coordinates to angular coordinates.
        
        Args:
            x, y: Pixel coordinates
            image_width, image_height: Image dimensions
            
        Returns:
            tuple: (azimuth, elevation) in degrees relative to camera center
        """
        # Get undistorted normalized coordinates
        x_norm, y_norm = self._undistort_coordinates(x, y, image_width, image_height)
        
        # Convert to angles using FOV
        azimuth = x_norm * (self.lens.horizontal_fov / 2)
        elevation = y_norm * (self.lens.vertical_fov / 2)
        
        return azimuth, elevation

    def detect_horizon(self, image: np.ndarray) -> Optional[float]:
        """
        Detect horizon line in image.
        
        Args:
            image: BGR image from webcam
            
        Returns:
            float: Horizon elevation in degrees or None if horizon not found
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Use Hough transform to detect lines
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None:
            return None
            
        # Find most horizontal line
        best_line = None
        min_angle_diff = float('inf')
        
        for rho, theta in lines[:, 0]:
            # Convert theta to angle from horizontal
            angle = np.degrees(theta) % 180
            angle_diff = min(abs(angle - 0), abs(angle - 180))
            
            if angle_diff < min_angle_diff:
                min_angle_diff = angle_diff
                best_line = (rho, theta)
                
        if best_line is None or min_angle_diff > 10:  # Max 10 degrees from horizontal
            return None
            
        # Convert rho, theta to y-position
        rho, theta = best_line
        height = image.shape[0]
        y = int(rho / np.cos(theta))
        
        # Convert final y position to elevation angle
        _, elevation = self._pixel_to_angles(image.shape[1]/2, y, 
                                              image.shape[1], image.shape[0])
        return elevation

    def detect_sun(self, image: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Detect sun position in image.
        
        Args:
            image: BGR image from webcam
            
        Returns:
            tuple: (azimuth, elevation) in degrees relative to camera center
                  or None if sun not found
        """
        # Convert to HSV for better brightness detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Threshold for very bright regions
        _, brightness = cv2.threshold(hsv[:,:,2], 250, 255, cv2.THRESH_BINARY)
        
        # Find contours of bright regions
        contours, _ = cv2.findContours(brightness, cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Find the most circular bright region within size constraints
        best_sun = None
        best_circularity = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter == 0:
                continue
                
            radius = np.sqrt(area / np.pi)
            if radius < self.min_sun_radius or radius > self.max_sun_radius:
                continue
                
            # Circularity = 4π(area/perimeter²), 1.0 = perfect circle
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > best_circularity:
                best_circularity = circularity
                best_sun = contour
                
        if best_sun is None or best_circularity < 0.8:  # Minimum circularity threshold
            return None
            
        # Get center of sun
        M = cv2.moments(best_sun)
        if M['m00'] == 0:
            return None
            
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        # Convert center position to angles
        azimuth, elevation = self._pixel_to_angles(cx, cy, 
                                                 image.shape[1], image.shape[0])
        
        return azimuth, elevation

    def get_sun_position(self, image: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Get absolute sun bearing and elevation from image.
        
        Args:
            image: BGR image from webcam
            
        Returns:
            tuple: (bearing, elevation) in degrees
                  or None if sun or horizon not detected
        """
        # Detect horizon and sun
        horizon_elevation = self.detect_horizon(image)
        sun_pos = self.detect_sun(image)
        
        if horizon_elevation is None or sun_pos is None:
            return None
            
        sun_azimuth, sun_elevation = sun_pos
        
        # Calculate absolute bearing
        bearing = (self.camera_bearing + sun_azimuth) % 360
        
        # Calculate elevation relative to horizon
        elevation = sun_elevation - horizon_elevation
        
        return bearing, elevation

def create_fisheye_lens(fov_degrees: float = 180.0) -> CameraLens:
    """Create a lens configuration for a fisheye camera."""
    return CameraLens(
        horizontal_fov=fov_degrees,
        vertical_fov=fov_degrees,
        k1=-0.3,  # Example fisheye distortion parameters
        k2=0.1
    )

def create_rectilinear_lens(horizontal_fov: float = 60.0, 
                           vertical_fov: float = None) -> CameraLens:
    """Create a lens configuration for a standard rectilinear camera."""
    if vertical_fov is None:
        vertical_fov = horizontal_fov * 0.75  # Typical 4:3 aspect ratio
    return CameraLens(
        horizontal_fov=horizontal_fov,
        vertical_fov=vertical_fov
    )

def process_webcam_image(image: np.ndarray,
                        camera_bearing: float,
                        lens: CameraLens = None) -> Optional[Tuple[float, float]]:
    """
    Convenience function to process a single webcam image.
    
    Args:
        image: BGR image from webcam
        camera_bearing: Bearing of camera center in degrees
        lens: CameraLens object or None (will create default rectilinear lens)
        
    Returns:
        tuple: (bearing, elevation) in degrees
               or None if sun or horizon not detected
    """
    if lens is None:
        lens = create_rectilinear_lens()
    
    detector = SunDetector(lens=lens, camera_bearing=camera_bearing)
    return detector.get_sun_position(image)

# Example usage:
# fisheye_lens = create_fisheye_lens(180.0)
# cap = cv2.VideoCapture(0)
# ret, frame = cap.read()
# if ret:
#     result = process_webcam_image(frame, camera_bearing=180.0, lens=fisheye_lens)
#     if result:
#         bearing, elevation = result
#         print(f"Sun bearing: {bearing:.1f}°, elevation: {elevation:.1f}°")
# cap.release()