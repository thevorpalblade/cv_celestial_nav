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

class SunDetectionError(Exception):
    """Base exception for sun detection errors."""
    pass

class HorizonNotFoundError(SunDetectionError):
    """Raised when horizon cannot be detected in image."""
    pass

class SunNotFoundError(SunDetectionError):
    """Raised when sun cannot be detected in image."""
    pass

class UnphysicalMeasurementError(SunDetectionError):
    """Raised when calculated angles exceed physical limitations of the camera."""
    pass

class SunDetector:
    def __init__(self, 
                 lens: CameraLens,
                 camera_bearing: float = 0.0,  # Bearing of camera center in degrees
                 min_sun_radius: Optional[int] = None,  # Minimum radius of sun in pixels
                 max_sun_radius: Optional[int] = None   # Maximum radius of sun in pixels
                ):
        self.lens = lens
        self.camera_bearing = camera_bearing
        self.min_sun_radius = min_sun_radius
        self.max_sun_radius = max_sun_radius
        
    def _undistort_coordinates(self, x: float, y: float, 
                             image_width: int, image_height: int) -> Tuple[float, float]:
        """
        Convert distorted image coordinates to undistorted normalized coordinates.
        
        Args:
            x, y: Pixel coordinates (y increases downward)
            image_width, image_height: Image dimensions
            
        Returns:
            tuple: (x, y) in normalized coordinates (-1 to 1)
               y is still in image coordinates (positive downward)
        """
        # Convert to normalized coordinates (-1 to 1)
        x_norm = (x - image_width/2) / (image_width/2)
        y_norm = (y - image_height/2) / (image_height/2)
        
        # Convert to camera coordinates
        x_cam = x_norm
        y_cam = y_norm
        
        # Calculate radius from center
        r2 = x_cam*x_cam + y_cam*y_cam
        
        # Apply radial distortion correction
        distortion_factor = 1 + self.lens.k1*r2 + self.lens.k2*r2*r2
        
        x_undist = x_cam * distortion_factor
        y_undist = y_cam * distortion_factor
        
        # Apply tangential distortion correction
        x_undist += (2*self.lens.p1*x_cam*y_cam + self.lens.p2*(r2 + 2*x_cam*x_cam))
        y_undist += (self.lens.p1*(r2 + 2*y_cam*y_cam) + 2*self.lens.p2*x_cam*y_cam)
        
        return x_undist, y_undist

    def _pixel_to_angles(self, x: float, y: float, 
                        image_width: int, image_height: int) -> Tuple[float, float]:
        """
        Convert pixel coordinates to angular coordinates.
        
        Args:
            x, y: Pixel coordinates (y increases downward)
            image_width, image_height: Image dimensions
            
        Returns:
            tuple: (azimuth, elevation) in degrees relative to camera center
                   positive elevation means upward from center
        """
        # Get undistorted normalized coordinates
        x_norm, y_norm = self._undistort_coordinates(x, y, image_width, image_height)
        
        # Convert to angles using FOV
        # For a rectilinear lens, the relationship between normalized coordinates
        # and angle is: tan(angle) = normalized_coord * tan(fov/2)
        azimuth = math.degrees(math.atan(x_norm * math.tan(math.radians(self.lens.horizontal_fov / 2))))
        
        # Negate y_norm because image coordinates increase downward but we want positive angles upward
        elevation = -math.degrees(math.atan(y_norm * math.tan(math.radians(self.lens.vertical_fov / 2))))
        
        return azimuth, elevation

    def _calculate_sun_radius_bounds(self, image_width: int) -> Tuple[int, int]:
        """
        Calculate expected minimum and maximum sun radius in pixels based on image size.
        
        Args:
            image_width: Width of the image in pixels
            
        Returns:
            tuple: (min_radius, max_radius) in pixels
        """
        # Calculate pixels per degree based on actual image width
        pixels_per_degree = image_width / self.lens.horizontal_fov
        # Sun's angular diameter is approximately 0.5 degrees
        expected_radius = (0.5 * pixels_per_degree) / 2
        
        min_radius = self.min_sun_radius if self.min_sun_radius is not None else int(expected_radius * 10)
        max_radius = self.max_sun_radius if self.max_sun_radius is not None else int(expected_radius * 50)
        
        return min_radius, max_radius

    def detect_horizon(self, image: np.ndarray, horizon=None) -> float:
        """
        Detect horizon line in image, allowing for non-horizontal orientations.
        
        Args:
            image: BGR image from webcam
            
        Returns:
            float: Horizon elevation in degrees
            
        Raises:
            HorizonNotFoundError: If no suitable horizon line is detected
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection with adjusted parameters for horizon detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Use Hough transform to detect lines
        # Reduced angle resolution for better line detection
        lines = cv2.HoughLines(edges, 1, np.pi/360, threshold=100)
        
        if lines is None:
            raise HorizonNotFoundError("No lines detected in image")
            
        # Find longest line that could be horizon
        best_line = None
        best_score = float('-inf')
        height, width = image.shape[:2]
        
        for rho, theta in lines[:, 0]:
            # Convert theta to angle from horizontal (-90 to 90 degrees)
            angle = np.degrees(theta) - 90
            if angle > 90:
                angle -= 180
            
            # Calculate line endpoints
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            
            # Calculate line endpoints at image boundaries
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            
            # Calculate line length within image bounds
            # Clip line endpoints to image boundaries
            x1 = max(0, min(width-1, x1))
            x2 = max(0, min(width-1, x2))
            y1 = max(0, min(height-1, y1))
            y2 = max(0, min(height-1, y2))
            
            line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            # Score based on line length and angle from horizontal
            # Prefer longer lines and those closer to horizontal
            angle_penalty = abs(angle) / 90.0  # 0 for horizontal, 1 for vertical
            score = line_length * (1.0 - angle_penalty)
            
            if score > best_score:
                best_score = score
                best_line = (rho, theta, x1, y1, x2, y2)
        
        if best_line is None:
            raise HorizonNotFoundError("No suitable horizon line found")
        if horizon:
            best_line = 0, 0, horizon[0], horizon[1], horizon[2], horizon[3]        
        rho, theta, x1, y1, x2, y2 = best_line
        
        # Calculate the midpoint of the line
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # Convert midpoint to elevation angle
        _, elevation = self._pixel_to_angles(mid_x, mid_y, width, height)
        
        # Store line parameters for visualization
        self._last_horizon_line = best_line
        
        return elevation

    def detect_sun(self, image: np.ndarray, visualize: bool = False, horizon=None) -> Tuple[float, float]:
        """
        Detect sun position in image.
        
        Args:
            image: BGR image from webcam
            visualize: If True, displays image with detection visualization
            
        Returns:
            tuple: (azimuth, elevation) in degrees relative to camera center
            
        Raises:
            SunNotFoundError: If no suitable sun circle is detected
        """
        # Make a copy for visualization
        vis_image = image.copy() if visualize else None
        
        # Calculate sun radius bounds based on actual image size
        min_radius, max_radius = self._calculate_sun_radius_bounds(image.shape[1])
        
        # Convert to HSV for better brightness detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Threshold for very bright regions
        _, brightness = cv2.threshold(hsv[:,:,2], 240, 255, cv2.THRESH_BINARY)
        
        # Find contours of bright regions
        contours, _ = cv2.findContours(brightness, cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise SunNotFoundError("No bright regions detected in image")
            
        # Find the most circular bright region within size constraints
        best_sun = None
        best_circularity = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter == 0:
                continue
                
            radius = np.sqrt(area / np.pi)
            if radius < min_radius or radius > max_radius:
                continue
                
            # Circularity = 4π(area/perimeter²), 1.0 = perfect circle
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > best_circularity:
                best_circularity = circularity
                best_sun = contour
                
        if best_sun is None:
            raise SunNotFoundError(
                f"No circular bright region found between {min_radius} and {max_radius} pixels radius"
            )
        
        if best_circularity < 0.2:  # Minimum circularity threshold
            raise SunNotFoundError(f"Best candidate sun region not circular enough (circularity = {best_circularity:.2f})")
            
        # Get center and bounding box of sun
        M = cv2.moments(best_sun)
        if M['m00'] == 0:
            raise SunNotFoundError("Could not compute center of sun region")
            
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        if visualize:
            # Draw bounding box around sun
            x, y, w, h = cv2.boundingRect(best_sun)
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(vis_image, (cx, cy), 3, (0, 0, 255), -1)
            
            try:
                # Draw horizon line
                horizon_elevation = self.detect_horizon(image, horizon)
                if hasattr(self, '_last_horizon_line'):
                    _, _, x1, y1, x2, y2 = self._last_horizon_line
                    cv2.line(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            except HorizonNotFoundError:
                pass  # Skip horizon line if not found
            
            # Display the image
            cv2.imshow('Sun Detection', vis_image)
            cv2.waitKey(1)  # Update display
        
        # Convert center position to angles
        azimuth, elevation = self._pixel_to_angles(cx, cy, 
                                                 image.shape[1], image.shape[0])
        
        return azimuth, elevation

    def get_sun_position(self, image: np.ndarray, horizon=None) -> Tuple[float, float]:
        """
        Get absolute sun bearing and elevation from image.
        
        Args:
            image: BGR image from webcam
            horizon: Optional pre-detected horizon line coordinates
            
        Returns:
            tuple: (bearing, elevation) in degrees, where:
                   - bearing is absolute compass bearing
                   - elevation is relative to horizon (positive = above horizon)
            
        Raises:
            HorizonNotFoundError: If horizon cannot be detected
            SunNotFoundError: If sun cannot be detected
            UnphysicalMeasurementError: If calculated angles exceed camera's physical limitations
        """
        # Detect horizon and sun
        horizon_elevation = self.detect_horizon(image, horizon)  # Relative to camera center
        sun_azimuth, sun_elevation = self.detect_sun(image, horizon=horizon)  # Relative to camera center
        
        # Calculate absolute bearing
        bearing = (self.camera_bearing + sun_azimuth) % 360
        
        # Calculate elevation relative to horizon
        # If sun_elevation > horizon_elevation, sun is above horizon
        relative_elevation = sun_elevation - horizon_elevation
        
        # Check if the calculated elevation is physically possible
        max_possible_elevation = self.lens.vertical_fov / 2
        if abs(relative_elevation) > max_possible_elevation:
            raise UnphysicalMeasurementError(
                f"Calculated elevation {relative_elevation:.1f}° exceeds camera's physical limit "
                f"of ±{max_possible_elevation:.1f}° relative to horizon"
            )
        
        return bearing, relative_elevation

def create_fisheye_lens(horizontal_fov: float = 180.0,
                        vertical_fov: float = None) -> CameraLens:
    """Create a lens configuration for a fisheye camera."""
    if vertical_fov is None:
        vertical_fov = horizontal_fov * 0.75  # Typical 4:3 aspect ratio    

    return CameraLens(
        horizontal_fov=horizontal_fov,
        vertical_fov=vertical_fov,
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
                        lens: CameraLens = None) -> Tuple[float, float]:
    """
    Convenience function to process a single webcam image.
    
    Args:
        image: BGR image as numpy array
        camera_bearing: Bearing of camera center in degrees
        lens: CameraLens object or None (will create default rectilinear lens)
        
    Returns:
        tuple: (bearing, elevation) in degrees
        
    Raises:
        HorizonNotFoundError: If horizon cannot be detected
        SunNotFoundError: If sun cannot be detected
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