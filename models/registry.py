"""Look up backend specs by stable name from registry.yaml."""
from pathlib import Path

REGISTRY_PATH = Path(__file__).parent / "registry.yaml"


def load_registry() -> dict:
    """Return the full {name: entry} mapping."""
    import yaml

    return yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))["models"]


def resolve(name: str) -> dict:
    """Return a backend spec dict (type, path) for the named base model."""
    entry = load_registry()[name]
    return {"type": entry["type"], "path": entry["weights_dir"]}
