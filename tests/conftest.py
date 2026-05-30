import pytest
import random
from pathlib import Path

from engine.graph import load_map
from engine.env import make_initial_state


@pytest.fixture(scope="session")
def gm():
    return load_map(Path("maps/whitechapel.json"))


@pytest.fixture
def initial_state(gm):
    return make_initial_state(gm, rng=random.Random(42))
