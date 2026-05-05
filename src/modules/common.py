import yaml
from pathlib import Path

def load_config(config_path=None):
    """Load YAML config with dynamic path resolution."""
    if config_path is None:
        # Get project root: modules/ -> project root
        module_dir = Path(__file__).parent
        project_root = module_dir.parent
        config_path = project_root / "configs" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_project_root():
    """Get absolute path to project root."""
    module_dir = Path(__file__).parent
    return module_dir.parent
