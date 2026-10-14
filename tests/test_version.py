"""The version of the radio is kept twice: in pyproject.toml and for the surfaces."""

import re
import tomllib
from pathlib import Path

from ui.receiver_controller import APPLICATION_VERSION

PYPROJECT = Path(__file__).parents[1] / "pyproject.toml"


def test_the_version_shown_is_the_one_of_the_project():
    with PYPROJECT.open("rb") as file:
        assert tomllib.load(file)["project"]["version"] == APPLICATION_VERSION


def test_the_version_has_three_numbers():
    assert re.fullmatch(r"\d+\.\d+\.\d+", APPLICATION_VERSION)
