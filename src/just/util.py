from typing import Iterable, TypeVar
import dataclasses as D

U = TypeVar("U")


Maybe = tuple[U] | tuple[()]


@D.dataclass
class first[U]:
    iterable: Iterable[U]

    def or_else(self, default: U) -> U:
        return next(iter(self.iterable), default)


def maybe[U](v: U | None) -> Maybe[U]:
    return () if v is None else (v,)


def head_or_none[U](i: Iterable[U]) -> U | None:
    return next(iter(i), None)
