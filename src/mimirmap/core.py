import cv2
import torch
import numpy as np
import math
import os
from scipy.spatial.transform import Rotation as R
from pyproj import Geod
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_midas_model():
    """Load MiDaS depth estimation model."""
    model_type = "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    midas.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()
    transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform
    return midas, transform

def estimate_depth(image_path, model, transform, auto_calibrate=True, camera_height=1.4, pitch_degrees=12.0, gopro_model="HERO8"):
    """
    Estimate depth with improved scaling to absolute metrics using ground plane calibration.
    
    Args:
        image_path: Path to the input image
        model: MiDaS model
        transform: MiDaS transform
        auto_calibrate: Whether to auto-calibrate the depth map using ground plane geometry
        camera_height: Height of the camera from the ground in meters
        pitch_degrees: Downward pitch of the camera in degrees
        gopro_model: GoPro camera model for FOV estimation
        
    Returns:
        img: Original image
        depth_map_meters: Depth map in meters
        reference_pixel: Reference point used for calibration (if auto_calibrate=True)
        reference_distance: Reference distance in meters (if auto_calibrate=True)
    """
    # Load and prepare image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(next(model.parameters()).device)

    # Generate depth prediction
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()

    # Handle negative depth values
    prediction = np.maximum(prediction, 0.1)  # Ensure positive depth with minimum threshold

    if auto_calibrate:
        # Get camera FOV information
        camera_fov = get_fov_from_camera_params(gopro_model)
        vertical_fov = camera_fov["vertical"]

        # Determine reference point for ground plane (visible road start)
        image_height = img.shape[0]
        image_width = img.shape[1]

        # Use a point approximately 3/4 down the image for the visible road
        ground_ref_v = int(image_height * 0.75)
        ground_ref_u = int(image_width / 2)  # Center horizontally

        # Calculate the reference distance using ground plane geometry
        reference_distance = calculate_ground_distance(
            ground_ref_v, image_height, camera_height, pitch_degrees, vertical_fov
        )

        # Get depth at the reference point
        relative_depth_at_reference = prediction[ground_ref_v, ground_ref_u]

        # Calculate scaling factor
        if relative_depth_at_reference > 0.1:
            depth_scale = reference_distance / relative_depth_at_reference
        else:
            # Try to find a better reference point nearby
            window_size = 25
            y_min = max(0, ground_ref_v - window_size)
            y_max = min(image_height, ground_ref_v + window_size + 1)
            x_min = max(0, ground_ref_u - window_size)
            x_max = min(image_width, ground_ref_u + window_size + 1)

            window = prediction[y_min:y_max, x_min:x_max]
            max_depth_idx = np.unravel_index(window.argmax(), window.shape)

            # Convert to image coordinates
            better_v = y_min + max_depth_idx[0]
            better_u = x_min + max_depth_idx[1]
            better_depth = prediction[better_v, better_u]

            # Calculate new reference distance
            new_reference_distance = calculate_ground_distance(
                better_v, image_height, camera_height, pitch_degrees, vertical_fov
            )

            depth_scale = new_reference_distance / better_depth
            
            # Update reference point
            ground_ref_u, ground_ref_v = better_u, better_v
            reference_distance = new_reference_distance

        # Apply scaling - convert to meters
        depth_map_meters = prediction * depth_scale

        # Apply a bilateral filter to reduce noise while preserving edges
        depth_map_meters = cv2.bilateralFilter(depth_map_meters.astype(np.float32),
                                              d=7, sigmaColor=0.1, sigmaSpace=5.0)

        reference_pixel = (ground_ref_u, ground_ref_v)
        return img, depth_map_meters, reference_pixel, reference_distance
    else:
        # Return the raw depth map without calibration
        return img, prediction, None, None

def get_depth_at_pixel(depth_map, x, y):
    """Get depth value at specific pixel coordinates."""
    if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
        return depth_map[y, x]
    else:
        raise ValueError(f"Pixel coordinates ({x}, {y}) are outside the image bounds: {depth_map.shape[1]}x{depth_map.shape[0]}")

def estimate_distance_from_depth(depth_value, scale=1.0):
    """Convert depth value to distance in meters."""
    return depth_value * scale

def calculate_depth_confidence(x, y, depth_map, window_size=5):
    """
    Calculate confidence in the depth estimate based on depth variation in the local neighborhood.
    Lower variation indicates higher confidence.

    Args:
        x, y: Pixel coordinates
        depth_map: Depth map
        window_size: Size of the neighborhood window to analyze

    Returns:
        confidence: 0-1 value representing confidence (1 = highest)
    """
    # Create a small window around the point
    y_min = max(0, y-window_size)
    y_max = min(depth_map.shape[0], y+window_size+1)
    x_min = max(0, x-window_size)
    x_max = min(depth_map.shape[1], x+window_size+1)

    local_region = depth_map[y_min:y_max, x_min:x_max]

    # Calculate coefficient of variation (std/mean) as a measure of consistency
    mean_depth = np.mean(local_region)
    if mean_depth > 0:
        std_depth = np.std(local_region)
        coeff_variation = std_depth / mean_depth

        # Convert to confidence (0-1 scale)
        # Lower variation = higher confidence
        confidence = max(0, min(1, 1 - coeff_variation))
    else:
        confidence = 0

    return confidence

def calculate_ground_distance(v, image_height, camera_height, pitch_deg, v_fov_deg):
    """
    Calculate the distance to a point on the ground plane using perspective geometry.

    Args:
        v: Vertical pixel coordinate (from top of image)
        image_height: Height of the image in pixels
        camera_height: Height of the camera from the ground in meters
        pitch_deg: Downward pitch of the camera in degrees
        v_fov_deg: Vertical field of view in degrees

    Returns:
        distance: Distance to the ground point in meters
    """
    # Calculate the angle for each pixel
    deg_per_pixel = v_fov_deg / image_height

    # Get center of image
    center_v = image_height / 2

    # Calculate angle from optical axis (negative for points below center)
    pixel_angle = (center_v - v) * deg_per_pixel

    # Total angle from horizontal
    total_angle_rad = math.radians(pitch_deg - pixel_angle)

    # Calculate distance using trigonometry (adjacent = opposite / tan(angle))
    if total_angle_rad > 0:  # Make sure we're looking downward
        distance = camera_height / math.tan(total_angle_rad)
        return distance
    else:
        return float('inf')  # Point is above horizon

def get_fov_from_camera_params(gopro_model):
    """Get the diagonal, horizontal, and vertical FOV for a camera model."""
    # Default FOV values for different GoPro models
    camera_fov = {
        "HERO8": {"diagonal": 80, "horizontal": 69.5, "vertical": 49.8},
        "HERO9": {"diagonal": 84, "horizontal": 73.6, "vertical": 53.4},
        "HERO10": {"diagonal": 84, "horizontal": 73.6, "vertical": 53.4},
        "HERO7": {"diagonal": 78, "horizontal": 66.9, "vertical": 45.8},
        "DEFAULT": {"diagonal": 80, "horizontal": 69.5, "vertical": 49.8}
    }

    return camera_fov.get(gopro_model.upper(), camera_fov["DEFAULT"])

def get_camera_parameters(model="HERO8", resolution=None):
    """
    Get generic camera parameters based on model and resolution.
    Values are approximate and should be replaced with actual calibration when possible.

    Args:
        model: Camera model (e.g., "HERO8")
        resolution: Tuple (width, height) of image resolution
        
    Returns:
        fx, fy, cx, cy (camera intrinsics)
    """
    if resolution is None:
        # Use a default resolution if none provided
        width, height = 2666, 2000
    else:
        width, height = resolution
        
    # Generic camera parameters
    camera_params = {
        "HERO8": {
            "focal_length_mm": 2.8,
            "sensor_width_mm": 6.17,
            "sensor_height_mm": 4.55,
            "fov_degrees": get_fov_from_camera_params("HERO8")["diagonal"],
        },
        "HERO9": {
            "focal_length_mm": 2.92,
            "sensor_width_mm": 6.20,
            "sensor_height_mm": 4.65,
            "fov_degrees": get_fov_from_camera_params("HERO9")["diagonal"],
        },
        "HERO10": {
            "focal_length_mm": 2.92,
            "sensor_width_mm": 6.20,
            "sensor_height_mm": 4.65,
            "fov_degrees": get_fov_from_camera_params("HERO10")["diagonal"],
        },
        # Default for unknown models
        "DEFAULT": {
            "focal_length_mm": 2.8,
            "sensor_width_mm": 6.0,
            "sensor_height_mm": 4.5,
            "fov_degrees": get_fov_from_camera_params("DEFAULT")["diagonal"],
        }
    }

    # Get parameters for the specified model or use DEFAULT
    params = camera_params.get(model.upper(), camera_params["DEFAULT"])

    # Calculate focal length in pixels using field of view
    fx = width / (2 * math.tan(math.radians(params["fov_degrees"] / 2)))
    fy = height / (2 * math.tan(math.radians(params["fov_degrees"] * height / width / 2)))

    # Principal point (usually at the center of the image)
    cx = width / 2
    cy = height / 2

    return fx, fy, cx, cy

def pixel_to_camera_coords(u, v, Z, fx, fy, cx, cy):
    """Convert pixel coordinates to camera coordinates."""
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z])

def apply_orientation(point, yaw, pitch, roll):
    """Apply camera orientation to the point in camera coordinates."""
    r = R.from_euler('ZYX', [yaw, pitch, roll], degrees=True)
    return r.apply(point)

def project_gps(lat, lon, bearing, distance_m, altitude_change=0):
    """
    Project a point from a starting GPS position along a bearing for a specified distance.
    
    Args:
        lat, lon: Starting coordinates in decimal degrees
        bearing: Direction in degrees (0=North, 90=East, etc.)
        distance_m: Distance in meters
        altitude_change: Change in altitude (vertical) in meters
        
    Returns:
        lat2, lon2: Projected coordinates in decimal degrees
        alt2: New altitude
    """
    geod = Geod(ellps='WGS84')
    lon2, lat2, _ = geod.fwd(lon, lat, bearing, distance_m)
    return lat2, lon2

def local_to_gps(offset, lat0, lon0, alt0):
    """
    Convert local ENU coordinates to GPS.
    
    Args:
        offset: Numpy array [east, up, north] in meters
        lat0, lon0, alt0: Reference GPS coordinates
        
    Returns:
        lat1, lon1, alt1: Resulting GPS coordinates
    """
    geod = Geod(ellps="WGS84")
    east, up, north = offset[0], offset[1], offset[2]
    horizontal_dist = np.hypot(east, north)
    azimuth = np.degrees(np.arctan2(east, north)) % 360
    lon1, lat1, _ = geod.fwd(lon0, lat0, azimuth, horizontal_dist)
    return lat1, lon1, alt0 + up

def annotate_image(img, point, label, color=(0, 255, 0), radius=10, font_scale=0.6, thickness=2):
    """
    Annotate an image with a point and label.
    
    Args:
        img: Input image
        point: (x, y) coordinates to mark
        label: Text label to add
        color: BGR color tuple
        radius: Circle radius
        font_scale: Size of text
        thickness: Line thickness
        
    Returns:
        Annotated image
    """
    annotated = img.copy()
    cv2.circle(annotated, point, radius, color, -1)
    cv2.putText(annotated, label, (point[0]+10, point[1]-10), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    return annotated

def save_depth_heatmap(depth_map, output_path, reference_pixel=None, reference_distance=None):
    """
    Save a heatmap visualization of the depth map.
    
    Args:
        depth_map: Depth map array
        output_path: Path to save the visualization
        reference_pixel: Optional reference pixel (x,y) to mark
        reference_distance: Optional reference distance to display
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(depth_map, cmap='plasma')
    plt.colorbar(label='Depth (meters)')
    
    if reference_pixel is not None:
        plt.scatter(reference_pixel[0], reference_pixel[1], c='white', s=50)
        if reference_distance is not None:
            plt.annotate(f"{reference_distance:.1f}m", 
                         (reference_pixel[0]+10, reference_pixel[1]-10),
                         color='white', fontsize=8)
    
    plt.title('Depth Map')
    plt.axis('off')
    plt.tight_layout()
    
    # Make sure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def visualize_3d_position(camera_coords, object_coords, reference_coords=None, output_path=None):
    """
    Visualize the 3D positioning between camera and object.
    
    Args:
        camera_coords: Camera position (typically origin)
        object_coords: Object position in 3D space
        reference_coords: Optional reference point coordinates
        output_path: Path to save the visualization
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot camera at origin
    ax.scatter([camera_coords[0]], [camera_coords[1]], [camera_coords[2]], 
               color='red', s=100, label='Camera')

    # Plot object
    ax.scatter([object_coords[0]], [object_coords[1]], [object_coords[2]],
               color='blue', s=100, label='Object')

    # Draw line from camera to object
    ax.plot([camera_coords[0], object_coords[0]], 
            [camera_coords[1], object_coords[1]], 
            [camera_coords[2], object_coords[2]], 'b--')

    # Plot reference point if provided
    if reference_coords is not None:
        ax.scatter([reference_coords[0]], [reference_coords[1]], [reference_coords[2]],
                   color='green', s=100, label='Reference')
        # Draw line from camera to reference
        ax.plot([camera_coords[0], reference_coords[0]], 
                [camera_coords[1], reference_coords[1]], 
                [camera_coords[2], reference_coords[2]], 'g--')

    # Set labels
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Up (m)')
    ax.set_title('3D Positioning')
    ax.legend()

    # Equal aspect ratio 
    max_range = max(1.0, np.max(np.abs([
        object_coords[0], object_coords[1], object_coords[2],
        0, 0, 0  # Camera is at origin
    ])))
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    
    plt.close()

def estimate_object_gps(
    image_path, x, y,
    camera_lat, camera_lon, camera_alt,
    camera_model="HERO8",
    yaw=0.0, pitch=12.0, roll=0.0,
    camera_height=1.4,
    model=None, transform=None,
    visualize=False
):
    """
    Estimate the GPS coordinates of an object in the image.
    
    Args:
        image_path: Path to the input image
        x, y: Pixel coordinates of the object
        camera_lat, camera_lon, camera_alt: Camera GPS coordinates
        camera_model: Camera model for intrinsics estimation
        yaw, pitch, roll: Camera orientation in degrees
        camera_height: Height of the camera from ground in meters
        model: Optional pre-loaded MiDaS model
        transform: Optional pre-loaded MiDaS transform
        visualize: Whether to create and return visualizations
        
    Returns:
        lat, lon, alt: Estimated GPS coordinates of the object
        confidence: Confidence score for the estimation (0-1)
        distance: Estimated distance in meters
        visualizations: Dict of visualization paths if visualize=True
    """
    # Load MiDaS model if not provided
    if model is None or transform is None:
        model, transform = load_midas_model()
    
    # Estimate depth with auto-calibration
    img, depth_map, reference_pixel, reference_distance = estimate_depth(
        image_path, model, transform,
        auto_calibrate=True,
        camera_height=camera_height,
        pitch_degrees=pitch,
        gopro_model=camera_model
    )
    
    # Get image dimensions
    image_height, image_width = img.shape[:2]
    resolution = (image_width, image_height)
    
    # Get camera intrinsics
    fx, fy, cx, cy = get_camera_parameters(camera_model, resolution)
    
    # Get camera FOV information
    camera_fov = get_fov_from_camera_params(camera_model)
    vertical_fov = camera_fov["vertical"]
    
    # Check if pixel coordinates are valid
    if not (0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]):
        raise ValueError(f"Pixel coordinates ({x}, {y}) are outside the image bounds")

    # Get depth at the specified pixel
    Z = depth_map[y, x]
    
    # Check if the depth is too small, try to find a better point nearby
    if Z < 0.5:  # If depth is very small, likely inaccurate
        # Find the point with highest depth in a small window
        window_size = 25
        y_min = max(0, y - window_size)
        y_max = min(depth_map.shape[0], y + window_size + 1)
        x_min = max(0, x - window_size)
        x_max = min(depth_map.shape[1], x + window_size + 1)

        window = depth_map[y_min:y_max, x_min:x_max]
        max_depth_idx = np.unravel_index(window.argmax(), window.shape)

        # Convert to image coordinates
        better_y = y_min + max_depth_idx[0]
        better_x = x_min + max_depth_idx[1]
        better_Z = depth_map[better_y, better_x]

        if better_Z > Z:
            x, y, Z = better_x, better_y, better_Z
    
    # Calculate confidence in the depth estimate
    confidence = calculate_depth_confidence(x, y, depth_map)
    
    # If we still have a very low depth, use a fallback based on the auto-calibration
    if Z < 0.5:
        # Estimate distance based on position in image using ground plane geometry
        fallback_distance = calculate_ground_distance(
            y, image_height, camera_height, pitch, vertical_fov
        )
        
        # Only use this if the point is likely on the ground plane
        if 0 < fallback_distance < 100:  # Reasonable range check
            Z = fallback_distance
        else:
            # Use reference distance as approximate value
            Z = reference_distance
    
    # Convert pixel to camera coordinates
    point_cam = pixel_to_camera_coords(x, y, Z, fx, fy, cx, cy)
    
    # Apply camera orientation
    point_world = apply_orientation(point_cam, yaw, -pitch, roll)  # negate pitch if Y is down
    
    # Convert to ENU coordinates (East-North-Up)
    point_world_ENU = np.array([point_world[0], -point_world[1], point_world[2]])  # flip Y to up
    
    # Calculate straight-line distance
    distance = np.linalg.norm(point_world_ENU)
    
    # Convert to GPS
    lat, lon, alt = local_to_gps(point_world_ENU, camera_lat, camera_lon, camera_alt)
    
    # Create visualizations if requested
    visualizations = {}
    if visualize:
        # Annotate the original image
        annotated_img = annotate_image(img, (x, y), f"{Z:.1f}m")
        annotated_path = os.path.splitext(image_path)[0] + "_annotated.jpg"
        cv2.imwrite(annotated_path, annotated_img)
        visualizations['annotated_image'] = annotated_path
        
        # Create depth heatmap
        heatmap_path = os.path.splitext(image_path)[0] + "_depth.jpg"
        save_depth_heatmap(depth_map, heatmap_path, (x, y), Z)
        visualizations['depth_map'] = heatmap_path
        
        # Create 3D visualization
        vis_3d_path = os.path.splitext(image_path)[0] + "_3d.jpg"
        camera_origin = np.array([0, 0, 0])
        visualize_3d_position(camera_origin, point_world_ENU, None, vis_3d_path)
        visualizations['3d_position'] = vis_3d_path
    
    return lat, lon, alt, confidence, distance, visualizations if visualize else None