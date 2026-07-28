from .camera import Camera
from .sensor import Sensor

# TactileMeshVisualizer requires open3d's gui module; importing lazily so the
# rest of the package works headless.
try:
    from .visualizer import TactileMeshVisualizer  # noqa: F401
except Exception:
    TactileMeshVisualizer = None  # type: ignore

# Legacy point-cloud Visualizer is preserved for backwards compatibility.
from .visualizer import Visualizer  # noqa: F401