""""Test package."""

import warnings

# NumPy is intentionally not a dependency; PyTorch emits a probe warning.
warnings.filterwarnings("ignore", message="Failed to initialize NumPy.*")

