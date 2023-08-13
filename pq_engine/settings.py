import importlib
import os
from typing import Any

import pkg_resources

installed_packages = [
    package_info.key for package_info in pkg_resources.working_set
]

atol = float(os.getenv("PQENGINE_ATOL", default="1.0e-5"))
rtol = float(os.getenv("PQENGINE_RTOL", default="1.0e-5"))
rounded_decimal = int(os.getenv("PQENGINE_ROUNDED_DECIMAL", default="5"))
xp: Any = (
    importlib.import_module("cupy")
    if "cupy-cuda" in ",".join(installed_packages)
    and bool(os.getenv("PQENGINE_USE_CUPY", default="True"))
    else importlib.import_module("numpy")
)
