import time
from functools import wraps
from typing import TYPE_CHECKING, Callable, Concatenate, ParamSpec, TypeVar

import tqdm

if TYPE_CHECKING:
    from .base import SpiceBase

T = TypeVar('T')
S = TypeVar('S')
P = ParamSpec('P')


def progress_decorator(
    fn: Callable[Concatenate[S, P], T]
) -> Callable[Concatenate[S, P], T]:
    # pylint: disable=protected-access
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
            try:
                self._progress_call_stack.pop()
            except IndexError:
                pass
            return out

    return decorated  # type: ignore


class ProgressHook:
    def __call__(self, progress: float, stack: list[str]) -> None:
        raise NotImplementedError


class CLIProgressHook(ProgressHook):
    """
    General progress hook to display progress of each decorated function individually.
    """

    def __init__(self, leave: bool | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.leave = leave
        self.bars: dict[tuple[str, ...], tqdm.tqdm] = {}

    def __call__(self, progress: float, stack: list[str]) -> None:
        key = tuple(stack)
        if key not in self.bars:
            try:
                desc = stack[-1]
            except IndexError:
                desc = ''
            self.bars[key] = tqdm.tqdm(
                total=100,
                desc=desc,
                unit='%',
                bar_format='{l_bar}{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
                leave=(len(stack) == 1) if self.leave is None else self.leave,
            )
        self.bars[key].update(progress * 100 - self.bars[key].n)
        if progress == 1:
            self.bars[key].close()


class TotalTimingProgressHook(ProgressHook):
    """
    Progress hook to log time spent in each function.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.start_times: dict[tuple[str, ...], float] = {}
        self.self_durations: dict[tuple[str, ...], float] = {}

    def __call__(self, progress: float, stack: list[str]) -> None:
        key = tuple(stack)
        if key not in self.start_times:
            self.start_times[key] = time.time()
        if progress == 1:
            full_diration = time.time() - self.start_times[key]
            self_duration = full_diration
            for k, v in self.self_durations.items():
                if len(k) > len(key) and k[: len(key)] == key:
                    self_duration -= v
            self.self_durations[key] = self_duration
            desc = '> ' * (len(key) - 1) + key[-1]
            print(f'{full_diration:5.2f} {self_duration:5.2f}  {desc}')


# Specific progress hooks for use in saving
class _SaveProgressHook(ProgressHook):
    """
    Base class for progress hook to use when saving data.

    The subclasses attempt to roughly work out the overall progress by using some
    very quick benchmarks I've manually thrown together (`weights`). These are not
    accurate at all, but do give a better idea of the progress level than just giving
    e.g. the number of backplanes generated. It could definitely be improved by e.g.
    calculating the fraction of the observation which is on-disc, but this is basically
    just cosmetic and progress bars are hard (https://xkcd.com/612/).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fudge_factor = 1.01
        self.total_progress = 0
        self.progress_parts: dict[str, float] = {}
        self.weights: dict[str, float] = {}
        self.overall_progress = 0
        self.default_key = ''

    def __call__(self, progress: float, stack: list[str]) -> None:
        if self.total_progress == 0:
            self.total_progress = sum(self.weights.values()) * self.fudge_factor
        try:
            key = stack[-1]
        except IndexError:
            key = self.default_key
        progress_change = (
            (progress - self.progress_parts.get(key, 0))
            * self.weights.get(key, 0)
            / self.total_progress
        )
        overall_progress = self.overall_progress + progress_change
        if overall_progress > self.total_progress:
            progress_change = self.total_progress - self.overall_progress
            self.overall_progress = self.total_progress
        else:
            self.overall_progress = overall_progress
        self.progress_parts[key] = progress
        self.update_bar(progress_change)

    def update_bar(self, progress_change: float) -> None:
        raise NotImplementedError

    def get_description(self) -> str:
        return ''


# pylint: disable-next=abstract-method
class _SaveNavProgressHook(_SaveProgressHook):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weights: dict[str, float] = {
            'BodyXY._get_targvec_img': 100,
            'BodyXY._get_lonlat_img': 50,
            'BodyXY._get_radec_img': 5,
            'BodyXY._get_lonlat_centric_img': 5,
            'BodyXY._get_illumination_gie_img': 50,
            'BodyXY._get_state_imgs': 30,
            'BodyXY.get_radial_velocity_img': 5,
            'BodyXY._get_ring_coordinate_imgs': 50,
            'Observation.save_observation': 100,
        }
        self.default_key = 'Observation.save_observation'

    def get_description(self) -> str:
        return 'Saving observation'


# pylint: disable-next=abstract-method
class _SaveMapProgressHook(_SaveProgressHook):
    def __init__(self, n_wavelengths: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_wavelengths = n_wavelengths
        self.weights: dict[str, float] = {
            'Observation._get_mapped_data': n_wavelengths * 5,
            'BodyXY._get_targvec_map': 10,
            'BodyXY._get_illumf_map': 5,
            'BodyXY._get_radec_map': 10,
            'BodyXY._get_xy_map': 10,
            # 'BodyXY._get_lonlat_map': 10,
            'BodyXY._get_lonlat_centric_map': 5,
            'BodyXY._get_state_maps': 5,
            'BodyXY._get_ring_coordinate_maps': 10,
            'Observation.save_mapped_observation': 20,
        }
        self.default_key = 'Observation.save_mapped_observation'

    def get_description(self) -> str:
        return 'Saving map'


class _SaveProgressHookCLI(_SaveProgressHook):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bar = tqdm.tqdm(
            total=100,
            desc=self.get_description(),
            unit='%',
            bar_format='{l_bar}{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
            leave=True,
        )

    def update_bar(self, progress_change: float) -> None:
        # * 0.99999 to ensure floating point errros don't accumulate to >100%
        self.bar.update(progress_change * 100 * 0.99999)
        if self.progress_parts.get(self.default_key, 0) >= 1:
            self.bar.update(100 - self.overall_progress * 100)
            self.bar.close()


class SaveNavProgressHookCLI(_SaveNavProgressHook, _SaveProgressHookCLI):
    pass


class SaveMapProgressHookCLI(_SaveMapProgressHook, _SaveProgressHookCLI):
    pass
