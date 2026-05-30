"""
Shared training-run logging (D6): a tqdm progress bar + periodic console lines + optional wandb.

Used by both the SFT and GRPO loops so the two phases share one UX. wandb is *optional*: it is only
used when installed AND configured (``WANDB_API_KEY`` / ``WANDB_MODE`` set, or ``report_to='wandb'``),
otherwise it degrades to a no-op. Each phase logs whatever metrics it wants (loss / IoU / reward
components); this class is metric-agnostic.
"""
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def wandb_enabled(report_to: str = 'auto') -> bool:
    """
    Whether to log to wandb. ``report_to``: ``'wandb'`` forces on (if importable), ``'none'`` forces
    off, ``'auto'`` (default) turns it on only when wandb is importable AND configured via env.
    """
    if report_to == 'none':
        return False
    try:
        import wandb  # noqa: F401
    except ImportError:
        if report_to == 'wandb':
            logger.warning('report_to="wandb" but wandb is not installed; logging to console only.')
        return False
    if report_to == 'wandb':
        return True
    return bool(os.environ.get('WANDB_API_KEY') or os.environ.get('WANDB_MODE'))


class RunLogger:
    """
    A thin progress-bar + metrics logger.

    :param total: Total number of steps (for the tqdm bar).
    :param phase: A short name shown on the bar / used as a metric prefix (e.g. ``'sft'``, ``'grpo'``).
    :param report_to: ``'auto'`` | ``'wandb'`` | ``'none'`` (see :func:`wandb_enabled`).
    :param console_every: Print a console line every N ``log`` calls (0 to disable console lines).
    :param wandb_kwargs: Extra kwargs forwarded to ``wandb.init`` (project, name, config, ...).
    """

    def __init__(
            self,
            total: int,
            phase: str = 'train',
            report_to: str = 'auto',
            console_every: int = 50,
            **wandb_kwargs: Any,
    ):
        self.phase = phase
        self.console_every = console_every
        self._n = 0

        from tqdm.auto import tqdm
        self.bar = tqdm(total=total, desc=phase, dynamic_ncols=True)

        self.use_wandb = wandb_enabled(report_to)
        self.wandb = None
        if self.use_wandb:
            import wandb
            self.wandb = wandb
            if wandb.run is None:
                wandb.init(**wandb_kwargs)

    def log(self, metrics: dict[str, float], step: int | None = None, advance: int = 1) -> None:
        """Logs a dict of scalars: updates the bar postfix, optional console line, and wandb."""
        self._n += 1
        if advance:
            self.bar.update(advance)
        postfix = {k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in metrics.items()}
        self.bar.set_postfix(postfix)

        if self.console_every and (self._n % self.console_every == 0):
            joined = ' '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}'
                              for k, v in metrics.items())
            self.bar.write(f'[{self.phase}] step {step if step is not None else self._n}  {joined}')

        if self.wandb is not None:
            self.wandb.log({f'{self.phase}/{k}': v for k, v in metrics.items()}, step=step)

    def log_image(self, key: str, image, step: int | None = None, caption: str | None = None) -> None:
        """Logs a PIL image (e.g. a visualize_sample contact sheet) to wandb; no-op otherwise."""
        if self.wandb is not None:
            self.wandb.log({f'{self.phase}/{key}': self.wandb.Image(image, caption=caption)}, step=step)

    def write(self, msg: str) -> None:
        """Prints a line above the progress bar (does not disturb it)."""
        self.bar.write(msg)

    def close(self) -> None:
        self.bar.close()
        if self.wandb is not None and self.wandb.run is not None:
            self.wandb.finish()
