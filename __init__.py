import importlib
import pkgutil

# On parcourt tous les sous-packages du dossier ImagesCameras/ImagesCameras
__all__ = []

for loader, module_name, is_pkg in pkgutil.iter_modules(__path__):
    # Import dynamique du module
    module = importlib.import_module(f"{__name__}.{module_name}")
    globals()[module_name] = module
    __all__.append(module_name)