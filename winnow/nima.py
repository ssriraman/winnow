"""Aesthetic scoring via a pretrained NIMA model (AVA-trained, through pyiqa).

Wraps `pyiqa <https://github.com/chaofengc/IQA-PyTorch>`_'s ``nima`` metric —
a NIMA model trained on the AVA aesthetic dataset — behind a small
``estimate(path) -> float`` interface. Unlike the earlier code, this loads
*real*, trained aesthetic weights rather than a randomly initialised head.

pyiqa (and its torch/torchvision stack) is an **optional** dependency, because
its weights carry a noncommercial license. Install it with::

    uv sync --extra aesthetic          # or: pip install 'winnow[aesthetic]'

Weights are downloaded and cached by pyiqa on first use (network required once,
into ``~/.cache/torch/hub/pyiqa``).
"""

_IMPORT_HINT = (
    "The aesthetic pass needs the optional 'aesthetic' extra (pyiqa + torch).\n"
    "Install it with:\n"
    "    uv sync --extra aesthetic\n"
    "  or\n"
    "    pip install 'winnow[aesthetic]'"
)

try:
    import torch
    from torchvision.transforms.functional import to_tensor
    import pyiqa
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(f"{_IMPORT_HINT}\n\n(underlying error: {exc})") from exc

from .config import NIMA_MAX_EDGE, NIMA_METRIC
from .io_utils import load_pil


class NimaEstimator:
    """Predicts a NIMA aesthetic score (~1-10, higher is better) for an image."""

    def __init__(self, device="cuda", metric=NIMA_METRIC, max_edge=NIMA_MAX_EDGE):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_edge = max_edge
        self.metric = pyiqa.create_metric(metric, device=self.device)
        # The rest of the pipeline treats a higher score as better; make sure the
        # chosen metric agrees rather than silently inverting the culling logic.
        if self.metric.lower_better:
            raise ValueError(
                f"pyiqa metric {metric!r} is lower-is-better; the culling logic "
                "assumes higher-is-better aesthetic scores."
            )

    def estimate(self, image_path) -> float:
        """Return the mean aesthetic score for a RAW or standard image path."""
        return self.estimate_image(load_pil(image_path))

    @torch.no_grad()
    def estimate_image(self, img) -> float:
        """Return the mean aesthetic score for an already-decoded PIL RGB image.

        Callers that also need the pixels (e.g. the dedupe pass computing a
        perceptual hash) decode once and pass the image here, avoiding a second
        RAW decode. The image may be down-scaled in place to bound VRAM."""
        if self.max_edge and max(img.size) > self.max_edge:
            img.thumbnail((self.max_edge, self.max_edge))
        tensor = to_tensor(img).unsqueeze(0).to(self.device)  # (1, 3, H, W) in [0, 1]
        return float(self.metric(tensor).item())

    def empty_cache(self):
        """Release cached CUDA memory between images (no-op on CPU)."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
