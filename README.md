# MimirMap

**MimirMap** is a Python library for estimating GPS coordinates of objects from a single monocular image using deep learning and geometric calculations. The library leverages the MiDaS depth estimation model and ground plane geometry to provide accurate distance and position estimation.

## Features

- **Depth Estimation**: Uses Intel's MiDaS model to estimate depth from a single image
- **Auto-Calibration**: Automatically calibrates depth maps using ground plane geometry
- **Distance Calculation**: Multiple methods to calculate the distance to objects
- **GPS Projection**: Converts pixel coordinates to real-world GPS coordinates
- **Confidence Metrics**: Provides confidence scores for depth and distance estimates
- **Visualizations**: Generates depth maps, 3D positioning visualizations, and annotated images
- **Camera Support**: Pre-configured for various GoPro models with customizable parameters
- **Interactive Mode**: Click on images to instantly get GPS coordinates
- **Batch Processing**: Process multiple points in one command

## Installation

```bash
# From PyPI (when published)
pip install mimirmap

# From source
git clone https://github.com/joaquinolivera/Mimirmap.git
cd mimirmap
pip install -e .
```

## Dependencies

- PyTorch
- OpenCV
- NumPy
- SciPy
- Matplotlib
- PyProj

## Basic Usage

### Command Line Interface

```bash
# Basic usage
mimirmap image.jpg 640 480 -34.636059 -58.706453 --output-dir ./output

# With more options
mimirmap image.jpg 640 480 -34.636059 -58.706453 \
  --altitude 50.0 \
  --camera-height 1.4 \
  --pitch 12.0 \
  --yaw 0.0 \
  --camera-model HERO8 \
  --method auto \
  --save-depth \
  --save-3d \
  --verbose
```

### Python API

```python
from mimirmap.core import load_midas_model, estimate_object_gps

# Load the MiDaS model
model, transform = load_midas_model()

# Estimate object GPS coordinates
lat, lon, alt, confidence, distance, visualizations = estimate_object_gps(
    image_path="path/to/image.jpg",
    x=640, y=480,  # Pixel coordinates of the object
    camera_lat=-34.636059, camera_lon=-58.706453, camera_alt=50.0,
    camera_model="HERO8",
    yaw=0.0, pitch=12.0, roll=0.0,
    camera_height=1.4,
    model=model, transform=transform,
    visualize=True
)

print(f"Estimated GPS: {lat}, {lon}, {alt}")
print(f"Estimated distance: {distance}m")
print(f"Confidence: {confidence}")
```

## Interactive Example

The library includes an example script demonstrating interactive use:

```bash
python -m examples.demo image.jpg --mode interactive
```

This opens a window where you can click on objects to get their estimated GPS coordinates.

## Advanced Usage

### Custom Camera Parameters

```python
from mimirmap.core import get_camera_parameters

# Get intrinsic parameters for a specific camera model and resolution
fx, fy, cx, cy = get_camera_parameters(
    model="HERO9",
    resolution=(3840, 2160)  # 4K resolution
)
```

### Depth Map Visualization

```python
from mimirmap.core import load_midas_model, estimate_depth, save_depth_heatmap

# Load model and estimate depth
model, transform = load_midas_model()
img, depth_map, ref_pixel, ref_distance = estimate_depth(
    "image.jpg", model, transform,
    auto_calibrate=True
)

# Save depth visualization
save_depth_heatmap(
    depth_map,
    "depth_visualization.jpg",
    reference_pixel=ref_pixel,
    reference_distance=ref_distance
)
```

### 3D Positioning Visualization

```python
from mimirmap.core import visualize_3d_position
import numpy as np

# Create a 3D visualization of camera and object position
camera_coords = np.array([0, 0, 0])  # Origin
object_coords = np.array([5, 10, -2])  # Object position in ENU coordinates

visualize_3d_position(
    camera_coords,
    object_coords,
    output_path="3d_position.jpg"
)
```

## How It Works

MimirMap combines deep learning-based depth estimation with geometric calculations to estimate the position of objects in 3D space:

1. **Depth Estimation**: Uses MiDaS to generate a depth map from a single image
2. **Auto-Calibration**: Uses ground plane geometry to calibrate the depth map to real-world scale
3. **3D Position**: Converts pixel coordinates and depth to 3D coordinates in camera space
4. **Orientation Adjustment**: Applies camera orientation (yaw, pitch, roll) to get world coordinates
5. **GPS Projection**: Projects the 3D position to GPS coordinates based on camera location

## Supported Camera Models

The library includes pre-configured parameters for several GoPro models:

- HERO7
- HERO8
- HERO9
- HERO10

For other cameras, you can use the default parameters or specify custom ones.

## Limitations

- Depth estimation accuracy depends on the MiDaS model's performance
- Best results are achieved with objects on the ground plane
- Camera height and orientation parameters should be accurate for best results
- Results may vary depending on lighting conditions and image quality

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this library in your research, please cite:

```
@software{mimirmap2025,
  author = Joaquin Olivera,
  title = {MimirMap: GPS Estimation from Monocular Images},
  year = {2025},
  url = {https://github.com/yourusername/mimirmap}
}
```
