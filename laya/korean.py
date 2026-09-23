"""Korean-first adapter for Laya.

This module intentionally stays thin: it pins Korean workloads to the multilingual
checkpoint while exposing a small backend contract that callers can keep even if
the underlying model is replaced later.
"""

from typing import Any, Dict, Optional, Protocol, Union

from .router import Router, normalise_name


State = Union[str, dict, list]


class DecisionEngine(Protocol):
    """Minimal interface expected by applications using a decision backend."""

    def predict(self, state: State, questions: Dict[str, Any]) -> Dict[str, Any]:
        ...


class KoreanDecisionEngine:
    """Run Korean decision workloads through a selected Laya checkpoint.

    The default is the multilingual checkpoint because upstream benchmarks show
    that the English checkpoint degrades sharply on Korean.

    Parameters
    ----------
    device:
        Torch device accepted by laya.Router, for example "cuda" or "cpu".
    token:
        Optional Hugging Face token.
    preload:
        Load the selected checkpoint immediately instead of on first prediction.
    model:
        Laya model name. Defaults to "multilingual".
    models:
        Optional Router model overrides, useful for local or fine-tuned checkpoints.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        token: Optional[str] = None,
        preload: bool = False,
        model: str = "multilingual",
        models: Optional[Dict[str, Any]] = None,
    ):
        self.model = normalise_name(model)
        self.router = Router(
            models=models,
            device=device,
            token=token,
            max_loaded=1,
            default=self.model,
        )
        if preload:
            self.router.preload([self.model])

    def predict(
        self,
        state: State,
        questions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate all questions in one forward pass."""
        return self.router.predict(
            state,
            questions,
            model=self.model,
            lang="ko",
        )

    system_one = predict
    judge = predict

    def route(self, state: State, questions: Optional[Dict[str, Any]] = None):
        """Return the pinned route without loading a model."""
        return self.router.route(
            state,
            questions or {},
            model=self.model,
            lang="ko",
        )

    @property
    def loaded(self):
        return self.router.loaded

    def unload(self):
        """Release the resident checkpoint."""
        self.router.unload(self.model)
