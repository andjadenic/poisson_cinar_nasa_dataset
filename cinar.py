"""Poisson combined INAR(p) models.

This module implements the CINAR(p) process introduced by Weiss (2008),

    X_t = D_{t,1} (alpha o_t X_{t-1}) + ...
          + D_{t,p} (alpha o_t X_{t-p}) + epsilon_t,

where D_t is multinomial with probabilities ``phi`` and the innovations are
Poisson distributed.  Both dependence structures from the paper are
available: repeated thinnings of the same observation can be independent or
identical.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.special import expit, logsumexp
from scipy.stats import binom, poisson


ThinningOperator = Literal["independent", "identical"]

__all__ = [
    "PoissonCINARp",
    "PoissonCINAR",
    "CINARp",
    "CINAR",
    "ThinningOperator",
]


class PoissonCINARp:
    """Poisson CINAR(p) model with independent or identical thinnings.

    Parameters
    ----------
    p:
        Autoregressive order.
    thinning_operator:
        ``"independent"`` redraws the thinning of an observation whenever it
        is selected. ``"identical"`` reuses one thinning result for all later
        selections of that observation.
    """

    def __init__(
        self,
        p: int,
        thinning_operator: ThinningOperator = "independent",
    ) -> None:
        self.p = p
        self.thinning_operator = thinning_operator

    def _unpack(self, transformed: np.ndarray) -> tuple[float, float, np.ndarray]:
        alpha = float(expit(transformed[0]))
        innovation_mean = float(np.exp(transformed[1]))
        if self.p == 1:
            phi = np.ones(1, dtype=np.float64)
        else:
            logits = np.append(transformed[2:], 0.0)
            logits -= np.max(logits)
            weights = np.exp(logits)
            phi = weights / np.sum(weights)
        return alpha, innovation_mean, phi

    def _pack_phi(self, phi: np.ndarray) -> np.ndarray:
        if self.p == 1:
            return np.empty(0, dtype=np.float64)
        return np.log(phi[:-1] / phi[-1])

    def _independent_log_likelihood(
        self,
        data: np.ndarray,
        alpha: float,
        innovation_mean: float,
        phi: np.ndarray,
    ) -> float:
        log_phi = np.log(phi)
        log_likelihood = 0.0

        for time in range(self.p, data.size):
            current = int(data[time])
            lag_log_probabilities = np.empty(self.p, dtype=np.float64)

            for lag in range(1, self.p + 1):
                source = int(data[time - lag])
                retained = np.arange(min(source, current) + 1)
                transition_terms = binom.logpmf(retained, source, alpha)
                transition_terms += poisson.logpmf(
                    current - retained, innovation_mean
                )
                lag_log_probabilities[lag - 1] = (
                    log_phi[lag - 1] + logsumexp(transition_terms)
                )

            log_likelihood += float(logsumexp(lag_log_probabilities))

        return log_likelihood

    @staticmethod
    def _initial_thinning_message(
        values: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        message = np.ones(tuple(int(value) + 1 for value in values))
        for axis, value in enumerate(values):
            retained = np.arange(int(value) + 1)
            probabilities = binom.pmf(retained, int(value), alpha)
            shape = [1] * values.size
            shape[axis] = probabilities.size
            message *= probabilities.reshape(shape)
        return message / np.sum(message)

    def _identical_factor_log_probabilities(
        self,
        state_values: np.ndarray,
        current: int,
        innovation_mean: float,
        phi: np.ndarray,
    ) -> np.ndarray:
        shape = tuple(int(value) + 1 for value in state_values)
        log_factor = np.full(shape, -np.inf, dtype=np.float64)

        for lag in range(1, self.p + 1):
            axis = self.p - lag
            retained = np.arange(int(state_values[axis]) + 1)
            component = np.log(phi[lag - 1]) + poisson.logpmf(
                current - retained, innovation_mean
            )
            component_shape = [1] * self.p
            component_shape[axis] = component.size
            log_factor = np.logaddexp(
                log_factor, component.reshape(component_shape)
            )

        return log_factor

    def _identical_filter(
        self,
        data: np.ndarray,
        alpha: float,
        innovation_mean: float,
        phi: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        state_values = data[: self.p].copy()
        message = self._initial_thinning_message(state_values, alpha)
        fitted = np.full(data.size, np.nan, dtype=np.float64)
        log_likelihood = 0.0

        for time in range(self.p, data.size):
            state_means = self._state_axis_means(message)
            fitted[time] = innovation_mean + float(phi @ state_means[::-1])

            log_factor = self._identical_factor_log_probabilities(
                state_values,
                int(data[time]),
                innovation_mean,
                phi,
            )
            factor_maximum = float(np.max(log_factor))
            weighted = message * np.exp(log_factor - factor_maximum)
            scaled_probability = float(np.sum(weighted))
            log_likelihood += factor_maximum + np.log(scaled_probability)
            posterior = weighted / scaled_probability

            if time < data.size - 1:
                posterior = np.sum(posterior, axis=0)
                new_value = int(data[time])
                new_thinning = binom.pmf(
                    np.arange(new_value + 1), new_value, alpha
                )
                message = posterior[..., None] * new_thinning
                state_values = np.append(state_values[1:], new_value)
            else:
                message = posterior

        return log_likelihood, fitted, message

    @staticmethod
    def _state_axis_means(message: np.ndarray) -> np.ndarray:
        means = np.empty(message.ndim, dtype=np.float64)
        for axis, size in enumerate(message.shape):
            other_axes = tuple(index for index in range(message.ndim) if index != axis)
            marginal = np.sum(message, axis=other_axes)
            means[axis] = float(np.arange(size) @ marginal)
        return means

    def _log_likelihood(
        self,
        data: np.ndarray,
        alpha: float,
        innovation_mean: float,
        phi: np.ndarray,
    ) -> float:
        if self.thinning_operator == "independent":
            return self._independent_log_likelihood(
                data, alpha, innovation_mean, phi
            )
        return self._identical_filter(
            data, alpha, innovation_mean, phi
        )[0]

    def _negative_log_likelihood(
        self,
        transformed: np.ndarray,
        data: np.ndarray,
    ) -> float:
        alpha, innovation_mean, phi = self._unpack(transformed)
        return -self._log_likelihood(data, alpha, innovation_mean, phi)

    def _starting_points(self, data: np.ndarray) -> list[np.ndarray]:
        centred = data.astype(np.float64) - np.mean(data)
        lag_one_denominator = np.sqrt(
            np.sum(centred[:-1] ** 2) * np.sum(centred[1:] ** 2)
        )
        lag_one_correlation = float(
            np.sum(centred[:-1] * centred[1:])
            / max(lag_one_denominator, np.finfo(np.float64).tiny)
        )
        moment_alpha = float(np.clip(lag_one_correlation, 0.05, 0.95))

        correlations = np.empty(self.p, dtype=np.float64)
        for lag in range(1, self.p + 1):
            denominator = np.sqrt(
                np.sum(centred[:-lag] ** 2)
                * np.sum(centred[lag:] ** 2)
            )
            correlations[lag - 1] = max(
                float(
                    np.sum(centred[:-lag] * centred[lag:])
                    / max(denominator, np.finfo(np.float64).tiny)
                ),
                0.0,
            )
        phi_from_correlations = (correlations + 0.05) / np.sum(
            correlations + 0.05
        )
        uniform_phi = np.full(self.p, 1.0 / self.p)

        starts = []
        for alpha, phi in (
            (moment_alpha, phi_from_correlations),
            (0.20, uniform_phi),
            (0.50, uniform_phi),
            (0.80, uniform_phi),
        ):
            innovation_mean = max((1.0 - alpha) * float(np.mean(data)), 0.05)
            starts.append(
                np.concatenate(
                    (
                        [np.log(alpha / (1.0 - alpha)), np.log(innovation_mean)],
                        self._pack_phi(phi),
                    )
                )
            )
        return starts

    def _independent_fitted_values(
        self,
        data: np.ndarray,
        alpha: float,
        innovation_mean: float,
        phi: np.ndarray,
    ) -> np.ndarray:
        fitted = np.full(data.size, np.nan, dtype=np.float64)
        for time in range(self.p, data.size):
            lags = data[time - self.p : time][::-1]
            fitted[time] = innovation_mean + alpha * float(phi @ lags)
        return fitted

    def fit(self, observations: Sequence[int] | np.ndarray) -> "PoissonCINARp":
        """Fit the model by conditional maximum likelihood.

        The fitted AIC, BIC and one-step prediction MSE are available as
        ``aic_``, ``bic_`` and ``mse_`` and together in ``metrics_``.
        """

        data = np.asarray(observations, dtype=np.int64)
        bounds = [(-12.0, 12.0), (-15.0, 15.0)] + [
            (-12.0, 12.0)
        ] * (self.p - 1)
        results: list[OptimizeResult] = []

        for start in self._starting_points(data):
            results.append(
                minimize(
                    self._negative_log_likelihood,
                    start,
                    args=(data,),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 1000, "ftol": 1.0e-11, "gtol": 1.0e-7},
                )
            )

        best = min(results, key=lambda result: float(result.fun))
        alpha, innovation_mean, phi = self._unpack(np.asarray(best.x))

        if self.thinning_operator == "independent":
            fitted = self._independent_fitted_values(
                data, alpha, innovation_mean, phi
            )
        else:
            _, fitted, _ = self._identical_filter(
                data, alpha, innovation_mean, phi
            )

        residuals = data[self.p :] - fitted[self.p :]
        parameter_count = self.p + 1
        effective_sample_size = data.size - self.p
        log_likelihood = -float(best.fun)

        self.data_ = data.copy()
        self.alpha_ = alpha
        self.innovation_mean_ = innovation_mean
        self.marginal_mean_ = innovation_mean / (1.0 - alpha)
        self.phi_ = phi
        self.parameters_ = {
            "alpha": alpha,
            "innovation_mean": innovation_mean,
            "marginal_mean": self.marginal_mean_,
            "phi": phi.copy(),
        }
        self.log_likelihood_ = log_likelihood
        self.aic_ = 2.0 * parameter_count - 2.0 * log_likelihood
        self.bic_ = float(
            np.log(effective_sample_size) * parameter_count
            - 2.0 * log_likelihood
        )
        self.mse_ = float(np.mean(residuals**2))
        self.metrics_ = {
            "AIC": self.aic_,
            "BIC": self.bic_,
            "MSE": self.mse_,
        }
        self.aic = self.aic_
        self.bic = self.bic_
        self.mse = self.mse_
        self.fitted_values_ = fitted
        self.residuals_ = residuals
        self.optimization_result_ = best
        return self

    def save_model(
        self,
        day: str,
        train_metrics: dict[str, float | int],
        validation_metrics: dict[str, float | int],
        test_metrics: dict[str, float | int],
        repository: str | Path = "saved_models",
    ) -> Path:
        """Save a fitted model and split-specific evaluation metrics as JSON.

        The output is ``<repository>/<day>.json``. Model parameters and fit
        statistics come from this fitted instance; train, validation, and test
        metrics describe the external model-selection evaluation workflow.
        """

        output_directory = Path(repository)
        output_directory.mkdir(parents=True, exist_ok=True)
        output_path = output_directory / f"{day}.json"

        def json_metrics(values: dict[str, float | int]) -> dict[str, float | int]:
            return {
                name: value.item() if isinstance(value, np.generic) else value
                for name, value in values.items()
            }

        saved_model = {
            "schema_version": 1,
            "day": day,
            "model": {
                "class": self.__class__.__name__,
                "p": self.p,
                "thinning_operator": self.thinning_operator,
            },
            "parameters": {
                "alpha": self.alpha_,
                "innovation_mean": self.innovation_mean_,
                "marginal_mean": self.marginal_mean_,
                "phi": self.phi_.tolist(),
            },
            "fit_statistics": {
                "log_likelihood": self.log_likelihood_,
                "AIC": self.aic_,
                "BIC": self.bic_,
                "MSE": self.mse_,
            },
            "evaluation_metrics": {
                "train": json_metrics(train_metrics),
                "validation": json_metrics(validation_metrics),
                "test": json_metrics(test_metrics),
            },
        }

        with output_path.open("w", encoding="utf-8") as stream:
            json.dump(saved_model, stream, indent=2)
            stream.write("\n")

        return output_path

    def _identical_last_thinning_means(self, data: np.ndarray) -> np.ndarray:
        state_values = data[: self.p].copy()
        message = self._initial_thinning_message(state_values, self.alpha_)

        for time in range(self.p, data.size):
            log_factor = self._identical_factor_log_probabilities(
                state_values,
                int(data[time]),
                self.innovation_mean_,
                self.phi_,
            )
            factor_maximum = float(np.max(log_factor))
            weighted = message * np.exp(log_factor - factor_maximum)
            posterior = weighted / np.sum(weighted)
            posterior = np.sum(posterior, axis=0)
            new_value = int(data[time])
            new_thinning = binom.pmf(
                np.arange(new_value + 1), new_value, self.alpha_
            )
            message = posterior[..., None] * new_thinning
            state_values = np.append(state_values[1:], new_value)

        return self._state_axis_means(message)

    def predict(
        self,
        observations: Sequence[int] | np.ndarray,
        n_steps: int | None = None,
    ) -> np.ndarray:
        """Predict future conditional means from an observed history.

        If ``n_steps`` is omitted, the horizon equals the number of supplied
        observations. Thus ``predict([x1, ..., xn])`` returns predictions for
        the next ``n`` time steps.
        """

        data = np.asarray(observations, dtype=np.int64)
        horizon = data.size if n_steps is None else n_steps

        if self.thinning_operator == "independent":
            thinning_means = (self.alpha_ * data[-self.p :]).tolist()
        else:
            thinning_means = self._identical_last_thinning_means(data).tolist()

        predictions = np.empty(horizon, dtype=np.float64)
        for step in range(horizon):
            prediction = self.innovation_mean_ + float(
                self.phi_ @ np.asarray(thinning_means[-self.p :][::-1])
            )
            predictions[step] = prediction
            thinning_means.append(self.alpha_ * prediction)

        return predictions

    def plot(
        self,
        observations: Sequence[int] | np.ndarray | None = None,
        n_steps: int = 0,
        ax: plt.Axes | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot observations, one-step model values, and optional forecasts."""

        data = self.data_ if observations is None else np.asarray(observations)
        if observations is None:
            fitted = self.fitted_values_
        elif self.thinning_operator == "independent":
            fitted = self._independent_fitted_values(
                data,
                self.alpha_,
                self.innovation_mean_,
                self.phi_,
            )
        else:
            _, fitted, _ = self._identical_filter(
                data,
                self.alpha_,
                self.innovation_mean_,
                self.phi_,
            )

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))
        figure = ax.figure
        time = np.arange(data.size)
        ax.plot(time, data, color="black", marker="o", markersize=3, label="Data")
        ax.plot(time, fitted, color="tab:blue", linewidth=2, label="CINAR fitted")

        if n_steps > 0:
            forecasts = self.predict(data, n_steps=n_steps)
            future_time = np.arange(data.size, data.size + n_steps)
            ax.plot(
                future_time,
                forecasts,
                color="tab:orange",
                marker="o",
                linestyle="--",
                label="Forecast",
            )

        ax.set_title(
            f"Poisson CINAR({self.p}) - {self.thinning_operator} thinnings"
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(alpha=0.25)
        figure.tight_layout()
        return figure, ax


PoissonCINAR = PoissonCINARp
CINARp = PoissonCINARp
CINAR = PoissonCINARp
