from typing import Iterable, TypeVar

U = TypeVar("U")


Maybe = tuple[U] | tuple[()]


def maybe[U](v: U | None) -> Maybe[U]:
    return () if v is None else (v,)


def head_or_none[U](i: Iterable[U]) -> U | None:
    return next(iter(i), None)
