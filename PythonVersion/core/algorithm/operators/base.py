"""Operator base class."""

from __future__ import annotations

from abc import ABC
from typing import ClassVar


class Operator(ABC):
    name: ClassVar[str]

