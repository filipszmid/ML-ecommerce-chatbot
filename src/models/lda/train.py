"""Linear Discriminant Analysis training workflow."""

from __future__ import annotations

from typing import Any

from hyperopt import hp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.models.common.base import BaseTrainWorkflow


class LDATrainWorkflow(BaseTrainWorkflow):
    """Train workflow for Linear Discriminant Analysis.

    The search space uses shrinkage with the lsqr/eigen solvers. This gives the
    LDA run a statistical strategy that is useful on small tabular datasets
    where covariance estimates can be noisy.
    """

    model_name = "lda"
    requires_scaling = True

    def default_params(self) -> dict[str, Any]:
        """Return default LDA parameters.

        Returns:
            Parameter dictionary.
        """
        return {
            "solver": "lsqr",
            "shrinkage": "auto",
        }

    def hyperopt_space(self) -> dict[str, Any]:
        """Return Hyperopt search space.

        Returns:
            Hyperopt space.
        """
        return {
            "solver": hp.choice("lda_solver", ["lsqr", "eigen"]),
            "shrinkage": hp.choice(
                "lda_shrinkage",
                ["auto", 0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0],
            ),
        }

    def build_estimator(self, params: dict[str, Any]) -> LinearDiscriminantAnalysis:
        """Build an LDA classifier.

        Args:
            params: Estimator parameters.

        Returns:
            LinearDiscriminantAnalysis.
        """
        return LinearDiscriminantAnalysis(**params)
