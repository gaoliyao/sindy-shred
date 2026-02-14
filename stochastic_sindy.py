"""Stochastic SINDy: sparse identification of stochastic differential equations.

Discovers dynamics of the form dX = mu(X)dt + sigma(X)dW from trajectory data,
where mu(X) is the drift and sigma(X) is the diffusion coefficient.

Primary method ('girsanov'):
    1. Recover sigma from quadratic variation + SINDy
    2. Compute Q-Brownian motion from sigma
    3. Extract signature coefficients (l21) via iisignature
    4. Recover mu from the level-2 iterated integral ODE
    5. Compute P-Brownian motion via Girsanov
    6. Refit mu as a sparse polynomial via SINDy (Route A)

Alternative method ('mle'):
    Joint maximum likelihood estimation. Parametrizes sigma(x) = exp(h(x))
    for guaranteed positivity and optimizes the Euler-Maruyama log-likelihood.
"""

import numpy as np
import pysindy as ps
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import minimize

try:
    import iisignature
    _HAS_IISIGNATURE = True
except ImportError:
    _HAS_IISIGNATURE = False


class StochasticSINDy:
    """Sparse identification of stochastic differential equations.

    Parameters
    ----------
    mu_degree : int
        Polynomial degree for drift basis functions.
    sigma_degree : int
        Polynomial degree for diffusion basis functions.
    method : str
        'girsanov' (signature-based, default) or 'mle' (maximum likelihood).
    sigma_threshold : float
        STLSQ sparsity threshold for sigma^2 discovery (girsanov method).
    sigma_alpha : float
        STLSQ ridge parameter for sigma^2 discovery (girsanov method).
    mu_threshold : float
        STLSQ sparsity threshold for mu discovery via Route A (girsanov method).
    mu_alpha : float
        STLSQ ridge parameter for mu discovery via Route A (girsanov method).
    signature_level : int
        Truncation level for iterated-integral signatures (girsanov method).
    ridge_alpha : float
        Ridge regression penalty for signature regression (girsanov method).
    mu_regularization : float
        L2 regularization on drift coefficients (mle method).
    sigma_regularization : float
        L2 regularization on diffusion coefficients (mle method).
    """

    def __init__(
        self,
        mu_degree=1,
        sigma_degree=1,
        method="girsanov",
        # Girsanov params
        sigma_threshold=0.015,
        sigma_alpha=100,
        mu_threshold=0.065,
        mu_alpha=0.05,
        signature_level=2,
        ridge_alpha=0.1,
        # MLE params
        mu_regularization=1e-2,
        sigma_regularization=1e-4,
    ):
        self.mu_degree = mu_degree
        self.sigma_degree = sigma_degree
        self.method = method
        # Girsanov
        self.sigma_threshold = sigma_threshold
        self.sigma_alpha = sigma_alpha
        self.mu_threshold = mu_threshold
        self.mu_alpha = mu_alpha
        self.signature_level = signature_level
        self.ridge_alpha = ridge_alpha
        # MLE
        self.mu_reg = mu_regularization
        self.sigma_reg = sigma_regularization

        # Fitted state (shared)
        self._d = None
        self.mse_ = None

        # Girsanov state (per dimension)
        self._sigma_models = None   # list of pysindy models for sigma^2
        self._mu_models = None      # list of pysindy models for mu (Route A)
        self.bq_ = None             # Q-Brownian motion per dim
        self.bp_ = None             # P-Brownian motion per dim
        self.l21_ = None            # signature coefficients per dim
        self.mu_trajectory_ = None  # mu(X_t) along trajectory per dim
        self.sigma_trajectory_ = None

        # MLE state
        self._mu_coeffs = None
        self._sigma_coeffs = None
        self._phi_mu = None
        self._phi_sigma = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, dt, mu_0=None):
        """Identify drift and diffusion from a trajectory.

        Parameters
        ----------
        X : array-like, shape (N,) or (N, d)
            Observed trajectory.
        dt : float
            Time step between observations.
        mu_0 : float or array-like, optional
            Initial drift value mu(X_0). If None, estimated from data.
            Only used by girsanov method.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self._d = X.shape[1]

        if self.method == "girsanov":
            if not _HAS_IISIGNATURE:
                raise ImportError(
                    "method='girsanov' requires iisignature. "
                    "Install with: pip install --no-build-isolation iisignature"
                )
            self._fit_girsanov(X, dt, mu_0)
        elif self.method == "mle":
            self._fit_mle(X, dt)
        else:
            raise ValueError(
                f"Unknown method '{self.method}', use 'girsanov' or 'mle'."
            )
        return self

    def predict_mu(self, x):
        """Evaluate the learned drift mu(x).

        Parameters
        ----------
        x : array-like

        Returns
        -------
        mu : ndarray (squeezed for 1-D systems).
        """
        x = self._to_2d(x)
        if self.method == "girsanov":
            result = np.column_stack([
                np.asarray(m.predict(x[:, i:i+1])).flatten()
                for i, m in enumerate(self._mu_models)
            ])
        else:
            result = self._phi_mu.transform(x) @ self._mu_coeffs
        return result.squeeze() if self._d == 1 else result

    def predict_sigma(self, x):
        """Evaluate the learned diffusion sigma(x).

        Parameters
        ----------
        x : array-like

        Returns
        -------
        sigma : ndarray (squeezed for 1-D systems).
        """
        x = self._to_2d(x)
        if self.method == "girsanov":
            result = np.zeros((x.shape[0], self._d))
            for i, m in enumerate(self._sigma_models):
                s2 = np.asarray(m.predict(x[:, i:i+1])).flatten()
                result[:, i] = np.sqrt(np.clip(s2, 1e-12, None))
        else:
            h = np.clip(self._phi_sigma.transform(x) @ self._sigma_coeffs, -50, 50)
            result = np.exp(h)
        return result.squeeze() if self._d == 1 else result

    def simulate(self, x0, t, noise=None, seed=None):
        """Euler-Maruyama forward simulation of the identified SDE.

        Parameters
        ----------
        x0 : array-like, shape (d,) or scalar
            Initial condition.
        t : array-like
            Time array (uniformly spaced).
        noise : array-like, optional
            Pre-generated noise increments dW.
        seed : int, optional
            Random seed.

        Returns
        -------
        X : ndarray, shape (len(t),) for 1-D or (len(t), d).
        """
        if seed is not None:
            np.random.seed(seed)

        t = np.asarray(t)
        dt = t[1] - t[0]
        x0 = np.atleast_1d(np.asarray(x0, dtype=float))
        d = len(x0)
        N = len(t)

        X = np.zeros((N, d))
        X[0] = x0

        if noise is None:
            dW = np.sqrt(dt) * np.random.randn(N - 1, d)
        else:
            dW = np.asarray(noise)
            if dW.ndim == 1:
                dW = dW.reshape(-1, 1)

        for i in range(N - 1):
            xi = X[i:i+1]
            mu = self.predict_mu(xi).reshape(d)
            sigma = self.predict_sigma(xi).reshape(d)
            X[i + 1] = X[i] + mu * dt + sigma * dW[i]

        return X.squeeze() if self._d == 1 else X

    def print_equations(self):
        """Print the discovered drift and diffusion equations."""
        if self.mse_ is not None:
            print(f"Reconstruction MSE: {self.mse_:.3e}")
        if self.method == "girsanov":
            for i in range(self._d):
                suffix = f"_{i}" if self._d > 1 else ""
                print(f"sigma{suffix}^2(X): ", end="")
                self._sigma_models[i].print()
                print(f"mu{suffix}(X): ", end="")
                self._mu_models[i].print()
        else:
            for i in range(self._d):
                suffix = f"_{i}" if self._d > 1 else ""
                mu_str = self._format_poly(self._mu_coeffs[:, i], self._phi_mu)
                sig_str = self._format_poly(self._sigma_coeffs[:, i], self._phi_sigma)
                print(f"mu{suffix}(X) = {mu_str}")
                print(f"log sigma{suffix}(X) = {sig_str}")
                print(f"  => sigma{suffix}(X) = exp(above)")

    # ------------------------------------------------------------------
    # Girsanov pipeline
    # ------------------------------------------------------------------

    def _fit_girsanov(self, X, dt, mu_0):
        """Full pipeline: QV → sigma → BQ → signatures → mu ODE → BP → Route A."""
        N, d = X.shape
        t = np.arange(N) * dt
        diff = ps.FiniteDifference(order=3)

        self._sigma_models = []
        self._mu_models = []
        self.bq_ = np.zeros((N, d))
        self.bp_ = np.zeros((N, d))
        self.l21_ = np.zeros((N, d))
        self.mu_trajectory_ = np.zeros((N, d))
        self.sigma_trajectory_ = np.zeros((N, d))

        for dim in range(d):
            S = X[:, dim]

            # ---- Step 1: sigma from quadratic variation ----
            QV = np.zeros(N)
            for i in range(1, N):
                QV[i] = QV[i - 1] + (S[i] - S[i - 1]) ** 2
            dQV = diff._differentiate(QV.reshape(-1, 1), t).flatten()

            sigma_model = ps.SINDy(
                optimizer=ps.STLSQ(
                    threshold=self.sigma_threshold, alpha=self.sigma_alpha
                ),
                feature_library=ps.PolynomialLibrary(degree=self.sigma_degree),
            )
            sigma_model.fit(S.reshape(-1, 1), t, x_dot=dQV.reshape(-1, 1))
            self._sigma_models.append(sigma_model)

            sigma_pred = np.sqrt(
                np.clip(np.asarray(sigma_model.predict(S.reshape(-1, 1))).flatten(),
                        1e-12, None)
            )
            self.sigma_trajectory_[:, dim] = sigma_pred

            # Sigma derivatives w.r.t. S (not t!)
            sigma_prime = diff._differentiate(
                sigma_pred.reshape(-1, 1), S
            ).flatten()
            sigma_2prime = diff._differentiate(
                sigma_prime.reshape(-1, 1), S
            ).flatten()

            # ---- Step 2: Q-Brownian motion ----
            BQ = np.zeros(N)
            for i in range(1, N):
                BQ[i] = BQ[i - 1] + (S[i] - S[i - 1]) / sigma_pred[i - 1]
            self.bq_[:, dim] = BQ

            # ---- Step 3: Signature coefficients l21 ----
            l21 = np.zeros(N)
            for j in range(N):
                path = np.column_stack([t[j:], BQ[j:]])
                sigs = np.array([
                    iisignature.sig(path[:i], self.signature_level)
                    for i in range(1, len(path) + 1)
                ])
                clf = Ridge(alpha=self.ridge_alpha)
                clf.fit(sigs, S[j:])
                l21[j] = clf.coef_[4]
            self.l21_[:, dim] = l21

            # ---- Step 4: mu from level-2 iterated integral ODE ----
            if mu_0 is not None:
                m0 = np.atleast_1d(mu_0)[dim] if np.ndim(mu_0) > 0 else float(mu_0)
            else:
                m0 = (S[1] - S[0]) / dt  # estimate from first increment

            sp = sigma_pred
            a = sigma_prime / sp
            b = (2 * l21 + sp * sigma_prime ** 2 + sp ** 2 * sigma_2prime) / (2 * sp)

            mu_pred = np.zeros(N)
            mu_pred[0] = m0
            for i in range(1, N):
                mu_pred[i] = (
                    mu_pred[i - 1]
                    + (a[i - 1] * mu_pred[i - 1] + b[i - 1]) * (S[i] - S[i - 1])
                )
            self.mu_trajectory_[:, dim] = mu_pred

            # ---- Step 5: P-Brownian motion ----
            BP = np.zeros(N)
            for i in range(1, N):
                BP[i] = (
                    BP[i - 1]
                    + BQ[i] - BQ[i - 1]
                    - (mu_pred[i - 1] / sigma_pred[i - 1]) * dt
                )
            self.bp_[:, dim] = BP

            # ---- Step 6: Refit mu as sparse polynomial (Route A) ----
            delta_BP = np.diff(BP)
            delta_X = np.diff(S)
            delta_sigma = np.diff(sigma_pred)

            mu_hat_points = (
                delta_X
                - sigma_pred[:-1] * delta_BP
                - 0.5 * sigma_pred[:-1] * delta_sigma * (delta_BP ** 2 - dt)
            ) / dt

            mu_model = ps.SINDy(
                optimizer=ps.STLSQ(
                    threshold=self.mu_threshold, alpha=self.mu_alpha
                ),
                feature_library=ps.PolynomialLibrary(degree=self.mu_degree, include_bias=False),
            )
            mu_model.fit(
                S[:-1].reshape(-1, 1), t[:-1], x_dot=mu_hat_points.reshape(-1, 1)
            )
            self._mu_models.append(mu_model)

        # Reconstruction MSE
        dX = np.diff(X, axis=0)
        mu_pred_all = np.column_stack([
            np.asarray(m.predict(X[:-1, i:i+1])).flatten()
            for i, m in enumerate(self._mu_models)
        ])
        self.mse_ = float(np.mean((dX - mu_pred_all * dt) ** 2))

    # ------------------------------------------------------------------
    # MLE internals
    # ------------------------------------------------------------------

    def _fit_mle(self, X, dt):
        N, d = X.shape
        dX = np.diff(X, axis=0)
        X_mid = X[:-1]

        self._phi_mu = PolynomialFeatures(degree=self.mu_degree, include_bias=True)
        self._phi_sigma = PolynomialFeatures(degree=self.sigma_degree, include_bias=True)

        PhiX_mu = self._phi_mu.fit_transform(X_mid)
        PhiX_sigma = self._phi_sigma.fit_transform(X_mid)
        p_mu = PhiX_mu.shape[1]
        p_sigma = PhiX_sigma.shape[1]

        self._mu_coeffs = np.zeros((p_mu, d))
        self._sigma_coeffs = np.zeros((p_sigma, d))

        for i in range(d):
            dX_i = dX[:, i]

            def nll(theta, _PhiMu=PhiX_mu, _PhiSg=PhiX_sigma, _dXi=dX_i, _dt=dt):
                a, b = theta[:p_mu], theta[p_mu:]
                mu_i = _PhiMu @ a
                h_i = np.clip(_PhiSg @ b, -50, 50)
                s2_i = np.exp(2.0 * h_i)
                resid = _dXi - mu_i * _dt
                val = 0.5 * np.sum(resid ** 2 / (s2_i * _dt) + np.log(s2_i * _dt))
                val += 0.5 * (self.mu_reg * np.sum(a ** 2) + self.sigma_reg * np.sum(b ** 2))
                return val

            theta0 = np.zeros(p_mu + p_sigma)
            res = minimize(nll, theta0, method="L-BFGS-B",
                           options=dict(maxiter=2000, ftol=1e-12))
            self._mu_coeffs[:, i] = res.x[:p_mu]
            self._sigma_coeffs[:, i] = res.x[p_mu:]

        mu_pred = PhiX_mu @ self._mu_coeffs
        self.mse_ = float(np.mean((dX - mu_pred * dt) ** 2))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_2d(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            x = x.reshape(1, 1)
        elif x.ndim == 1:
            if self._d == 1:
                x = x.reshape(-1, 1)
            else:
                x = x.reshape(1, -1)
        return x

    @staticmethod
    def _format_poly(coeffs, poly):
        names = poly.get_feature_names_out()
        terms = []
        for c, name in zip(coeffs, names):
            if abs(c) > 1e-10:
                terms.append(f"{c:.6g}*{name}")
        return " + ".join(terms) if terms else "0"
