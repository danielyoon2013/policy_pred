"""Policy catalog loading."""
from pathlib import Path


def load_catalog() -> list[dict]:
    """Load policies/catalog.yaml. Returns the 'policies' list."""
    import yaml

    from .. import config
    text = Path(config.POLICY_CATALOG_PATH).read_text(encoding="utf-8")
    return yaml.safe_load(text)["policies"]
