import importlib
import os
from typing import Any

import pkg_resources

installed_packages = [
    package_info.key for package_info in pkg_resources.working_set
]

#: atol value used by numpy or cupy.
atol = float(os.getenv("PQENGINE_ATOL", default="1.0e-5"))

#: rtol value used by numpy or cupy.
rtol = float(os.getenv("PQENGINE_RTOL", default="1.0e-5"))

#: approximation order of decimal used by numpy or cupy
rounded_decimal = int(os.getenv("PQENGINE_ROUNDED_DECIMAL", default="8"))

#: calculation engine (numpy or cupy)
xp: Any = (
    importlib.import_module("cupy")
    if "cupy-cuda" in ",".join(installed_packages)
    and bool(os.getenv("PQENGINE_USE_GPU", default="True"))
    else importlib.import_module("numpy")
)
