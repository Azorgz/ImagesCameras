import inspect
import re

from . import Metrics
from .Metrics import BaseMetric


def _class_to_key(name: str) -> str:
    """
    Naming rules:

    - If name length <= 5 → lowercase full name
    - If name length > 5:
        * extract consecutive uppercase groups
        * separate groups with "_"
        * lowercase result
    """

    if len(name) <= 5:
        return name.lower()

    # Extract blocks of consecutive uppercase letters
    groups = re.findall(r"[A-Z]+", name)
    n_groups = []
    for g in groups:
        if len(g) == 1:
            if n_groups and len(n_groups[-1]) == 1:
                n_groups[-1] += g
            else:
                n_groups.append(g)
        else:
            n_groups.append(g)
    return "_".join(n_groups).lower()


# Collect metric classes from Metrics module
_metrics_classes = [
    cls
    for _, cls in inspect.getmembers(Metrics, inspect.isclass)
    if issubclass(cls, BaseMetric)
    and cls is not BaseMetric
]

# Sort alphabetically (optional but cleaner)
_metrics_classes = sorted(_metrics_classes, key=lambda x: x.__name__)

__all__ = [cls.__name__ for cls in _metrics_classes]

METRICS_DICT = {
    _class_to_key(cls.__name__): cls
    for cls in _metrics_classes
}