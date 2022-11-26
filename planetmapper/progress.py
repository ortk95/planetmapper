from functools import wraps
from typing import TypeVar, ParamSpec, Concatenate, Callable, TYPE_CHECKING
import tqdm

if TYPE_CHECKING:
    from .base import SpiceBase

T = TypeVar('T')
S = TypeVar('S')
P = ParamSpec('P')


def progress_decorator(
    fn: Callable[Concatenate[S, P], T]
) -> Callable[Concatenate[S, P], T]:
    @wraps(fn)
    def decorated(self: 'SpiceBase', *args: P.args, **kwargs: P.kwargs):
        if self._progress_hook is None:
            return fn(self, *args, **kwargs)  # type: ignore
        else:
            self._progress_call_stack.append(fn.__qualname__)
            self._update_progress_hook(0)
            try:
                out = fn(self, *args, **kwargs)  # type: ignore
            except:
                self._progress_call_stack.pop()
                raise
            self._update_progress_hook(1)
            self._progress_call_stack.pop()
            return out

    return decorated  # type: ignore


class ProgressHook:
    def __call__(self, progress: float, stack: list[str]) -> None:
        raise NotImplementedError


class TqdmProgressHook(ProgressHook):
    def __init__(self) -> None:
        self.bars: dict[tuple[str, ...], tqdm.tqdm] = {}

    def __call__(self, progress: float, stack: list[str]) -> None:
        key = tuple(stack)
        if key not in self.bars:
            self.bars[key] = tqdm.tqdm(
                total=100,
                desc=stack[-1],
                unit='%',
                bar_format='{l_bar}{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
                leave=True,
            )
        self.bars[key].update(progress * 100 - self.bars[key].n)
        if progress == 1:
            self.bars[key].close()
