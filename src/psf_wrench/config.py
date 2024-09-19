"""
This file defines the configuration for an PSF Wrench run. It is currently
quite complex relative to the functionality, but is designed for
fine-grained control of future extensions.
"""

import glob
from pathlib import Path
import tosholi
import dataclasses
from typing import Optional
from dacite.exceptions import MissingValueError


@dataclasses.dataclass
class Config:
    """
    This class contains sensible defaults for a basic PSF Model.
    """

    version: str
    oversampling: Optional[float] = 10.0
    stamp_size: Optional[int] = 31

    # min and max object sizes for the PSF model
    min_object_size: Optional[int] = 5
    max_object_size: Optional[int] = 100

    # matching parameters for initial detection
    kernel_size: Optional[int] = 5


def get_config() -> Optional[Config]:
    """Grab the first TOML file in the CWD that looks like an argussim config."""
    tomls = glob.glob("*.toml")

    conf_found = False
    for toml in tomls:
        with open(toml, "rb") as f:
            try:
                config = tosholi.load(Config, f)
            except MissingValueError:
                continue
            conf_found = True
            break
    if conf_found:
        return config
    return None


def write_config(config: Config) -> Path:
    ""
    out_path = "psfwrench.toml"
    with open(out_path, "wb") as f:
        tosholi.dump(config, f)
    return Path(out_path)
