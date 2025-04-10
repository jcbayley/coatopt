import importlib.resources as pkg_resources
import configparser
import logging
from typing import Any
import ast
import json

logger = logging.getLogger(__name__)


class CoatingConfigParser(configparser.ConfigParser):
    """Config parser for bayesbeat"""

    default_config = pkg_resources.files("coatopt.config") / "default.ini"

    def __init__(self, *args, scheduler=None, **kwargs):
        super().__init__(*args, **kwargs)
        logger.debug(f"Loading default config from: {self.default_config}")
        self.read(self.default_config)

    def get(self, section, option, **kwargs):
        return try_literal_eval(super().get(section, option, **kwargs))

    def write_to_file(self, filename: str) -> None:
        """Write the config to a file"""
        with open(filename, "w") as f:
            self.write(f)


def try_literal_eval(value: Any, /) -> Any:
    """Try to call literal eval return value if an error is raised"""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value

def read_config(config_file: str, **kwargs) -> CoatingConfigParser:
    """Read a config file"""
    config = CoatingConfigParser(**kwargs)
    logger.info(f"Loading config from: {config_file}")
    config.read(config_file)
    return config


def read_materials(materials_file: str) -> dict:
    """Read a materials file"""
    if materials_file == "default":
        materials_file = pkg_resources.files("coatopt.config") / "materials.json"
    if materials_file == "default2":
        materials_file = pkg_resources.files("coatopt.config") / "materials2.json"
    elif materials_file == "default_noair":
        materials_file = pkg_resources.files("coatopt.config") / "materials_noair.json"
        
    with open(materials_file, "r") as f:
        materials = json.load(f)

    materials = {int(key): value for key, value in materials.items()}

    return materials