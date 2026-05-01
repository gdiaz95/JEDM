#!/usr/bin/env python3
"""NPGC (Non-Parametric Gaussian Copula) synthesizer.

This module defines the public ``NPGC`` class and its supporting private
helpers. The implementation models marginals with empirical CDFs (ECDFs)
and couples them through a Gaussian copula.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

__all__ = ["NPGC_local"]

LOGGER = logging.getLogger(__name__)

class NPGC_local:
    """Non-parametric Gaussian copula synthesizer for tabular data."""

    def __init__(
        self,
        enforce_min_max_values: bool = True,
        epsilon: float | None = 1.0,
        marginal_scale: bool = True,   # True → b = p/ε_m,          False → b = 1/ε_m
        corr_formula: bool = True,     # True → explicit R̂_{jk},    False → z_df.corr()
        corr_noise: bool = True,       # True → Φ⁻¹(n/n+1)²/(n-1)ε, False → 2/(n·ε)
        corr_delta: bool = True,       # True → 2√p/(n·ε_c),         False → 1e-8
    ) -> None:
        self.enforce_min_max_values = enforce_min_max_values
        self.epsilon = epsilon
        self.marginal_scale = marginal_scale
        self.corr_formula   = corr_formula
        self.corr_noise     = corr_noise
        self.corr_delta     = corr_delta
        self._model_state = {}
        self._fitted = False
        self._corr_delta = 1e-8   # overwritten at fit time when corr_delta=True and ε_c is available
        
    def fit(self, data: pd.DataFrame, epsilon: float | None = None, random_state: int | None = None) -> None:
        """Fit the synthesizer to a pandas DataFrame.
        
        Args:
            data: Input DataFrame to fit.
            epsilon: Differential privacy budget. Overrides instance epsilon if provided.
            random_state: Seed for reproducible DP noise. None means non-deterministic.
        """
        self._validate_data(data)
        eps = epsilon if epsilon is not None else self.epsilon
        LOGGER.info(f"Fitting with differential privacy epsilon={eps}...")
        LOGGER.info("Fitting NPGC...")
        
        # Learn marginals and transform to Gaussian space (Z).
        self._model_state = self._learn_distributions_and_correlation(data, epsilon=eps, random_state=random_state)
        self._fitted = True
        LOGGER.info("Model fitted successfully.")

    def sample(self, num_rows: int, seed: int | None = None) -> pd.DataFrame:
        """Sample synthetic rows from the fitted model."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call `fit` first.")
            
        LOGGER.info(f"Sampling {num_rows} rows...")
        return self._generate_samples(num_rows, seed)

    # Persistence helpers

    def save(self, filepath: str | os.PathLike[str]) -> None:
        """Serialize this fitted model to a pickle file path."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Cannot save.")

        # Create directory if needed
        path = Path(filepath)
        if path.parent and str(path.parent) != ".":
            path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("wb") as f:
            pickle.dump(self, f)

        LOGGER.info(f"Model saved to {filepath}")

    def load(self, filepath: str | os.PathLike[str]) -> None:
        """Load model state from a pickle file into this instance."""
        path = Path(filepath)
        with path.open("rb") as f:
            loaded_instance = pickle.load(f)
        
        if isinstance(loaded_instance, dict):
            # Fallback for older dictionary-based checkpoints
            self.__dict__.update(loaded_instance)
        else:
            # Standard behavior: loaded_instance is a full object
            self.__dict__.update(loaded_instance.__dict__)
            
        self._fitted = True
        LOGGER.info(f"Model loaded from {filepath}")

    # ============================================================
    # Internal Logic
    # ============================================================

    def _validate_data(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")
        if data.empty:
            raise ValueError("Data is empty.")

    def _learn_distributions_and_correlation(self, df: pd.DataFrame, epsilon: float | None = None, random_state: int | None = None) -> dict[str, Any]:
        """
        Learns the marginal distributions and correlation matrix with DP budget splitting.

        The total epsilon budget is split equally: half for marginals, half for the
        correlation matrix. This is a standard composition approach under DP.
        """
        if epsilon is None or epsilon <= 0:
            eps_marginal = None
            eps_corr = None
        else:
            # Equal budget split: eps/2 for marginals, eps/2 for correlation.
            eps_marginal = epsilon * 0.5
            eps_corr = epsilon * 0.5
        
        rng = np.random.default_rng(random_state)
        z_df = pd.DataFrame(index=df.index, columns=df.columns)
        marginals = {}
        n_cols = len(df.columns)   # number of features — used in marginal noise scale b = n_cols / ε_m

        for col in df.columns:
            series = df[col]
            valid_data = series.dropna()
            nan_frac = series.isna().mean()
            dtype = series.dtype
            if pd.api.types.is_datetime64_any_dtype(dtype):
                tz = getattr(dtype, "tz", None)
                ref = pd.Timestamp("1970-01-01", tz="UTC") if tz is not None else pd.Timestamp("1970-01-01")

                float_secs_valid = (valid_data - ref).dt.total_seconds().values
                float_secs_full = (series - ref).dt.total_seconds().values
                n_valid = len(float_secs_valid)

                if n_valid > 0 and eps_marginal is not None and eps_marginal > 0:
                    counts, edges = np.histogram(float_secs_valid, bins=100)
                    if edges[0] == edges[-1]:
                        sorted_values = np.full(n_valid, float(edges[0]))
                    else:
                        noise = rng.laplace(0, (n_cols if self.marginal_scale else 1.0) / eps_marginal, size=len(counts))
                        noisy_counts = np.maximum(counts.astype(float) + noise, 0.0)
                        total = noisy_counts.sum()
                        p = noisy_counts / total if total > 0 else np.ones(len(counts)) / len(counts)
                        bin_idx = rng.choice(len(counts), size=n_valid, replace=True, p=p)
                        dp_samples = edges[bin_idx] + rng.random(n_valid) * (edges[bin_idx + 1] - edges[bin_idx])
                        sorted_values = np.sort(dp_samples)
                else:
                    sorted_values = np.sort(float_secs_valid) if n_valid > 0 else np.array([])

                marginals[col] = {
                    "type": "datetime",
                    "sorted_values": sorted_values,
                    "nan_frac": nan_frac,
                    "dtype": dtype,
                    "dtype_name": str(dtype),
                    "tz": str(tz),
                }

                u = self._empirical_cdf_datetime_floats(float_secs_full, rng, epsilon=eps_marginal)

            elif pd.api.types.is_numeric_dtype(dtype):
                is_integer = np.allclose(valid_data % 1, 0) if len(valid_data) > 0 else False

                n_valid = len(valid_data)
                vals = valid_data.values.astype(float)

                if eps_marginal is not None and eps_marginal > 0 and n_valid > 0:
                    if is_integer:
                        # DP integer anchors via noisy counts on unique support
                        uniques, counts = np.unique(vals, return_counts=True)
                        counts = counts.astype(float)

                        noise = rng.laplace(0, (n_cols if self.marginal_scale else 1.0) / eps_marginal, size=len(counts))
                        noisy_counts = np.maximum(counts + noise, 1e-5)

                        p = noisy_counts / noisy_counts.sum()
                        dp_samples = rng.choice(uniques, size=n_valid, replace=True, p=p)
                        sorted_values = np.sort(dp_samples)

                        marginals[col] = {
                            "type": "integer",
                            "sorted_values": sorted_values,
                            "nan_frac": nan_frac,
                            "dtype": dtype,
                            "dtype_name": str(dtype),
                        }
                    else:
                        # DP continuous anchors via noisy histogram resampling
                        counts, edges = np.histogram(vals, bins=100)

                        # Handle constant column safely
                        if edges[0] == edges[-1]:
                            sorted_values = np.sort(np.full(n_valid, edges[0], dtype=float))
                        else:
                            noise = rng.laplace(0, (n_cols if self.marginal_scale else 1.0) / eps_marginal, size=len(counts))
                            noisy_counts = np.maximum(counts.astype(float) + noise, 0.0)

                            total = noisy_counts.sum()
                            if total <= 0:
                                # fallback: uniform over bins if noise zeroed everything
                                p = np.ones_like(noisy_counts) / len(noisy_counts)
                            else:
                                p = noisy_counts / total

                            bin_idx = rng.choice(len(counts), size=n_valid, replace=True, p=p)
                            left = edges[bin_idx]
                            right = edges[bin_idx + 1]
                            dp_samples = left + rng.random(n_valid) * (right - left)
                            sorted_values = np.sort(dp_samples)

                        marginals[col] = {
                            "type": "continuous",
                            "sorted_values": sorted_values,
                            "nan_frac": nan_frac,
                            "dtype": dtype,
                            "dtype_name": str(dtype),
                        }

                else:
                    # Non-DP: store raw empirical anchors
                    marginals[col] = {
                        "type": "integer" if is_integer else "continuous",
                        "sorted_values": np.sort(vals),
                        "nan_frac": nan_frac,
                        "dtype": dtype,
                        "dtype_name": str(dtype),
                    }

                u = self._empirical_cdf_continuous(series, rng, epsilon=eps_marginal)

            else:
                sorted_labels = np.sort(valid_data.unique()).tolist()
                val_counts = valid_data.value_counts()
                counts_in_order = np.array([val_counts.get(l, 0) for l in sorted_labels], dtype=float)

                if eps_marginal is not None and eps_marginal > 0:
                    noise = rng.laplace(0, (n_cols if self.marginal_scale else 1.0) / eps_marginal, size=len(counts_in_order))
                    counts_in_order = np.maximum(counts_in_order + noise, 1e-5)

                marginals[col] = {
                    "type": "categorical",
                    "labels": sorted_labels,            # NO "<NaN>"
                    "counts": counts_in_order.tolist(), # NO "<NaN>"
                    "nan_frac": nan_frac,
                    "dtype": dtype,
                    "dtype_name": str(dtype),
                }

                u = self._empirical_cdf_categorical(series, sorted_labels, rng, epsilon=eps_marginal)

            z_df[col] = self._uniform_to_gaussian(u)
            
        if self.corr_formula:
            Z = z_df.to_numpy(dtype=float)
            Z_c = Z - np.nanmean(Z, axis=0)
            s = np.nanstd(Z, axis=0, ddof=1)
            n = Z.shape[0]
            corr_vals = (Z_c.T @ Z_c) / ((n - 1) * np.outer(s, s))
            corr_vals = np.clip(corr_vals, -1.0, 1.0)
            np.fill_diagonal(corr_vals, 1.0)
            correlation_matrix = pd.DataFrame(corr_vals, columns=z_df.columns, index=z_df.columns).fillna(0.0)
        else:
            correlation_matrix = z_df.corr(method='pearson').fillna(0.0)

        if eps_corr is not None and eps_corr > 0:
            n = len(z_df)
            if self.corr_delta:
                self._corr_delta = 2 * np.sqrt(n_cols) / (n * eps_corr)
            noise_scale = (
                norm.ppf(n / (n + 1)) ** 2 / ((n - 1) * eps_corr)
                if self.corr_noise else
                2.0 / (n * eps_corr)
            )
            noise = rng.laplace(0, noise_scale, size=correlation_matrix.shape)
            noisy_corr = correlation_matrix + noise
            noisy_corr = (noisy_corr + noisy_corr.T) / 2
            noisy_vals = np.array(noisy_corr.to_numpy(), copy=True)  # guaranteed writable
            np.fill_diagonal(noisy_vals, 1.0)
            fixed_corr_values = self._get_nearest_correlation_matrix(noisy_vals)
            final_corr_df = pd.DataFrame(fixed_corr_values, columns=df.columns, index=df.columns)
        else:
            final_corr_df = correlation_matrix  
        
        return {
            "correlation_matrix": final_corr_df,
            "marginals": marginals,
            "columns": df.columns.tolist()
        }
    
    def _get_nearest_correlation_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Finds the nearest Positive Semi-Definite (PSD) matrix.
        Ensures the matrix can be used for Cholesky decomposition.
        """
        vals, vecs = np.linalg.eigh(matrix)
        vals = np.clip(vals, self._corr_delta, None)
        new_mat = (vecs * vals) @ vecs.T
        d = np.sqrt(np.diag(new_mat))
        new_mat = new_mat / np.outer(d, d)
        
        return new_mat

    def _generate_samples(self, n_samples: int, seed: int | None = None) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        
        state = self._model_state
        columns = state['columns']
        marginals = state['marginals']
        corr_matrix = state['correlation_matrix']

        # 1. Generate Independent Z
        z_ind = rng.standard_normal(size=(n_samples, len(columns)))
        z_ind_df = pd.DataFrame(z_ind, columns=columns)

        # 2. Apply Correlation
        z_correlated = self._apply_correlation(z_ind_df, corr_matrix)

        # 3. Inverse Transform
        synthetic_data = pd.DataFrame(index=range(n_samples), columns=columns)
        
        for col in columns:
            meta = marginals[col]
            u_samples = self._gaussian_to_uniform(z_correlated[col])
            
            if meta['type'] == 'continuous':
                synthetic_data[col] = self._inverse_ecdf_continuous(u_samples, meta)
            elif meta['type'] == 'integer':
                synthetic_data[col] = self._inverse_ecdf_integer(u_samples, meta)
            elif meta['type'] == 'categorical':
                synthetic_data[col] = self._inverse_ecdf_categorical(u_samples, meta)
            elif meta['type'] == 'datetime':
                synthetic_data[col] = self._inverse_ecdf_datetime(u_samples, meta)
                continue  # dtype already restored inside _inverse_ecdf_datetime

            # Restore Dtypes
            try:
                dtype_name = meta.get('dtype_name', '')
                if 'int' in dtype_name.lower() or meta['type'] == 'integer':
                    synthetic_data[col] = synthetic_data[col].round().astype(meta['dtype'])
                else:
                    synthetic_data[col] = synthetic_data[col].astype(meta['dtype'])
            except Exception as e:
                LOGGER.warning(f"Could not cast column '{col}': {e}")

        return synthetic_data

    # ============================================================
    # Math Helper Methods
    # ============================================================
    
    def _apply_correlation(self, z_ind: pd.DataFrame, corr_matrix: pd.DataFrame) -> pd.DataFrame:
        try:
            L = np.linalg.cholesky(corr_matrix.values)
        except np.linalg.LinAlgError:
            vals, vecs = np.linalg.eigh(corr_matrix.values)
            vals = np.clip(vals, self._corr_delta, None)
            R_pd = (vecs * vals) @ vecs.T
            D = np.sqrt(np.diag(R_pd))
            R_pd = R_pd / np.outer(D, D)
            L = np.linalg.cholesky(R_pd)
            
        X_arr = z_ind.values @ L.T
        return pd.DataFrame(X_arr, columns=z_ind.columns)

    def _gaussian_to_uniform(self, z: np.ndarray | pd.Series) -> np.ndarray:
        return norm.cdf(np.asarray(z, float))

    def _uniform_to_gaussian(self, u: np.ndarray | pd.Series) -> np.ndarray:
        u = np.asarray(u, float)
        u = np.clip(u, 1e-12, 1 - 1e-12)
        return norm.ppf(u)
    
    def _empirical_cdf_continuous(self, column: pd.Series, rng: np.random.Generator, epsilon: float | None = 1.0, integer_tolerance: float = 1e-12) -> np.ndarray:
        arr = np.asarray(column, float)
        mask = ~np.isnan(arr)
        valid = arr[mask]
        if valid.size == 0:
            return np.full(arr.shape, np.nan)

        is_int = np.allclose(valid, np.round(valid), atol=integer_tolerance)
        u = np.full(arr.shape, np.nan)

        if is_int:
            uniques, counts = np.unique(valid, return_counts=True)
            counts = counts.astype(float)
            if epsilon is not None and epsilon > 0:
                noise = rng.laplace(0, 1.0 / epsilon, size=len(counts))
                counts = np.maximum(counts + noise, 1e-5)
            
            p = counts / counts.sum()
            P = np.cumsum(p)
            L = np.insert(P[:-1], 0, 0)
            
            idx_map = {val: i for i, val in enumerate(uniques)}
            indices = np.array([idx_map[v] for v in valid])
            u[mask] = L[indices] + rng.random(valid.size) * p[indices]
        else:
            if epsilon is not None and epsilon > 0:
                # DP Path: Noisy Histograms provide a bounded-sensitivity ECDF
                counts, edges = np.histogram(valid, bins=100)
                if edges[0] == edges[-1]:
                    u[mask] = rng.random(valid.size)
                    return np.clip(u, 1e-12, 1 - 1e-12)
                noisy_counts = np.maximum(counts + rng.laplace(0, 1/epsilon, size=100), 0.0)
                total = noisy_counts.sum()

                if total <= 0:
                    u[mask] = rng.random(valid.size)
                else:
                    p = noisy_counts / total
                    P = np.cumsum(p)
                    L = np.insert(P[:-1], 0, 0.0)

                    bin_idx = np.searchsorted(edges, valid, side="right") - 1
                    bin_idx = np.clip(bin_idx, 0, len(p) - 1)

                    u[mask] = L[bin_idx] + rng.random(valid.size) * p[bin_idx]

                # FIX: replaced print() with LOGGER.debug() to avoid leaking DP internals to stdout
                if epsilon < 0.01:
                    LOGGER.debug(f"NPGC: Raw counts sum: {counts.sum()}")
                    LOGGER.debug(f"NPGC: Noisy counts sum: {noisy_counts.sum()}")
            else:
                # Standard Path: randomized PIT (ties -> uniform; constant column -> uniform)
                r_min = rankdata(valid, method="min")
                r_max = rankdata(valid, method="max")
                v = rng.random(valid.size)
                u[mask] = (r_min - 1 + v * (r_max - r_min + 1)) / valid.size

        return np.clip(u, 1e-12, 1 - 1e-12)
        
    def _empirical_cdf_categorical(
        self,
        column: pd.Series,
        sorted_labels: list[Any],
        rng: np.random.Generator,
        epsilon: float | None = 1.0,
    ) -> np.ndarray:
        """
        Categorical ECDF that is consistent with _inverse_ecdf_categorical:

        - meta['labels']/meta['counts'] exclude NaN
        - meta['nan_frac'] controls missingness mass
        - Non-missing categories occupy [0, 1 - nan_frac)
        - Missing values map into (1 - nan_frac, 1)
        """
        arr = np.asarray(column, dtype=object)
        is_nan = pd.isna(arr)
        n = arr.shape[0]

        u = np.full(n, np.nan, dtype=float)
        nan_frac = float(np.mean(is_nan))

        if np.all(is_nan):
            return np.clip(u, 1e-12, 1 - 1e-12)

        vals = arr[~is_nan]
        if vals.size == 0:
            return np.clip(u, 1e-12, 1 - 1e-12)

        uniq, cnt = np.unique(vals, return_counts=True)
        count_map = dict(zip(uniq, cnt))
        counts = np.array([count_map.get(l, 0) for l in sorted_labels], dtype=float)

        if epsilon is not None and epsilon > 0:
            noise = rng.laplace(0, 1.0 / epsilon, size=len(counts))
            counts = np.maximum(counts + noise, 1e-5)

        total = counts.sum()
        if total <= 0:
            p = np.ones_like(counts) / len(counts)
        else:
            p = counts / total

        P = np.cumsum(p)
        L = np.insert(P[:-1], 0, 0.0)

        idx_map = {lab: i for i, lab in enumerate(sorted_labels)}
        indices = np.array([idx_map[v] for v in vals], dtype=int)

        u_nonmiss_unit = L[indices] + rng.random(vals.size) * p[indices]
        u[~is_nan] = u_nonmiss_unit * max(1.0 - nan_frac, 0.0)

        if nan_frac > 0:
            u[is_nan] = 1.0 - rng.random(is_nan.sum()) * nan_frac

        return np.clip(u, 1e-12, 1 - 1e-12)

    def _inverse_ecdf_integer(self, u_values: np.ndarray | pd.Series, meta: dict[str, Any]) -> np.ndarray:
        sorted_vals = meta['sorted_values']
        nan_frac = meta['nan_frac']
        u = np.asarray(u_values, float)
        n = len(sorted_vals)
        result = np.full(u.shape, np.nan, dtype=float)
        if n == 0:
            return result
        mask_nan = np.isnan(u)
        result[mask_nan] = np.nan

        if nan_frac > 0:
            mask_cat = (~mask_nan) & (u > 1.0 - nan_frac)
            result[mask_cat] = np.nan
        else:
            mask_cat = np.zeros_like(u, bool)

        mask_valid = ~(mask_nan | mask_cat)
        u_valid = u[mask_valid]
        denom = max(1.0 - nan_frac, 1e-12)
        u_adj = np.clip(u_valid / denom, 0, 1 - 1e-8)
        knots_u = (np.arange(1, n+1) - 0.5) / n
        x_cont = self._interp_with_optional_extrapolation(u_adj, knots_u, sorted_vals)
        
        if self.enforce_min_max_values:
            uniques = np.unique(sorted_vals)
            diffs = np.abs(x_cont[:, None] - uniques[None, :])
            result[mask_valid] = uniques[diffs.argmin(axis=1)]
        else:
            result[mask_valid] = x_cont
        return result

    def _inverse_ecdf_categorical(self, u_values: np.ndarray | pd.Series, meta: dict[str, Any]) -> np.ndarray:
        sorted_labels = list(meta['labels'])
        counts = list(meta['counts'])
        nan_frac = meta['nan_frac']
        u = np.asarray(u_values, float)
        out = np.full(u.shape, np.nan, dtype=object)
        if nan_frac >= 1.0 - 1e-12:
            return out
        
        if nan_frac > 0:
            n_nan = sum(counts) * nan_frac / (1.0 - nan_frac) if sum(counts) > 0 else 0
            sorted_labels.append("<NaN>")
            counts.append(n_nan)

        mask_valid = ~np.isnan(u)
        if not np.any(mask_valid):
            return out

        u_adj = np.clip(u[mask_valid], 0.0, 1.0 - 1e-12)
        counts_arr = np.asarray(counts, float)
        if counts_arr.sum() <= 0:
            return out
        P = np.cumsum(counts_arr / counts_arr.sum())
        inds = np.clip(np.searchsorted(P, u_adj, side="right"), 0, len(sorted_labels) - 1)
        chosen = np.asarray(sorted_labels, dtype=object)[inds]
        chosen[chosen == "<NaN>"] = np.nan
        out[mask_valid] = chosen
        return out

    def _inverse_ecdf_continuous(self, u_values: np.ndarray | pd.Series, meta: dict[str, Any]) -> np.ndarray:
        sorted_vals = meta['sorted_values']
        nan_frac = meta['nan_frac']
        u = np.asarray(u_values, float)
        n = len(sorted_vals)
        result = np.full(u.shape, np.nan, dtype=float)
        if n == 0:
            return result
        mask_nan = np.isnan(u)
        result[mask_nan] = np.nan

        if nan_frac > 0:
            mask_cat = (~mask_nan) & (u > 1.0 - nan_frac)
            result[mask_cat] = np.nan
        else:
            mask_cat = np.zeros_like(u, bool)

        mask_valid = ~(mask_nan | mask_cat)
        denom = max(1.0 - nan_frac, 1e-12)
        u_adj = u[mask_valid] / denom
        if self.enforce_min_max_values:
            u_adj = np.clip(u_adj, 1e-8, 1 - 1e-8)
        knots_u = (np.arange(1, n+1) - 0.5) / n
        result[mask_valid] = self._interp_with_optional_extrapolation(u_adj, knots_u, sorted_vals)
        return result

    def _interp_with_optional_extrapolation(self, u: np.ndarray, knots_u: np.ndarray, sorted_vals: np.ndarray) -> np.ndarray:
        """Interpolate inverse ECDF with optional linear tail extrapolation."""
        x = np.interp(u, knots_u, sorted_vals)
        if self.enforce_min_max_values or len(sorted_vals) < 2:
            return x

        left = u < knots_u[0]
        right = u > knots_u[-1]
        if np.any(left):
            left_slope = (sorted_vals[1] - sorted_vals[0]) / max(knots_u[1] - knots_u[0], 1e-12)
            x[left] = sorted_vals[0] + (u[left] - knots_u[0]) * left_slope
        if np.any(right):
            right_slope = (sorted_vals[-1] - sorted_vals[-2]) / max(knots_u[-1] - knots_u[-2], 1e-12)
            x[right] = sorted_vals[-1] + (u[right] - knots_u[-1]) * right_slope
        return x

    def _empirical_cdf_datetime_floats(self, float_secs: np.ndarray, rng: np.random.Generator, epsilon: float | None = None) -> np.ndarray:
        """CDF for datetime columns on pre-converted float-seconds array. Always uses histogram path."""
        arr = np.asarray(float_secs, float)
        mask = ~np.isnan(arr)
        valid = arr[mask]
        u = np.full(arr.shape, np.nan)

        if valid.size == 0:
            return np.clip(u, 1e-12, 1 - 1e-12)

        if epsilon is not None and epsilon > 0:
            counts, edges = np.histogram(valid, bins=100)
            if edges[0] == edges[-1]:
                u[mask] = rng.random(valid.size)
            else:
                noisy_counts = np.maximum(counts.astype(float) + rng.laplace(0, 1.0 / epsilon, size=100), 0.0)
                total = noisy_counts.sum()
                if total <= 0:
                    u[mask] = rng.random(valid.size)
                else:
                    p = noisy_counts / total
                    P = np.cumsum(p)
                    L = np.insert(P[:-1], 0, 0.0)
                    bin_idx = np.searchsorted(edges, valid, side="right") - 1
                    bin_idx = np.clip(bin_idx, 0, len(p) - 1)
                    u[mask] = L[bin_idx] + rng.random(valid.size) * p[bin_idx]
        else:
            r_min = rankdata(valid, method="min")
            r_max = rankdata(valid, method="max")
            v = rng.random(valid.size)
            u[mask] = (r_min - 1 + v * (r_max - r_min + 1)) / valid.size

        return np.clip(u, 1e-12, 1 - 1e-12)

    def _inverse_ecdf_datetime(self, u_values: np.ndarray | pd.Series, meta: dict[str, Any]) -> pd.DatetimeIndex:
        """Inverse ECDF for datetime: maps u → float seconds → datetime."""
        float_secs = self._inverse_ecdf_continuous(u_values, meta)
        tz_str = meta.get("tz", "None")
        utc = tz_str not in (None, "None", "")
        result = pd.to_datetime(float_secs, unit="s", utc=utc)
        if utc:
            result = result.tz_convert(tz_str)
        try:
            result = result.astype(meta["dtype"])
        except Exception:
            pass
        return result
