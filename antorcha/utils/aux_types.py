from typing import TypeVar, Union

T = TypeVar('T')

maybe_pair = Union[T, tuple[T, T]]
square_2d = maybe_pair

