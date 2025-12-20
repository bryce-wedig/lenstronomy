# -*- coding: utf-8 -*-
"""
Multi-Gaussian Expansion (MGE) Multi-Set Light Profile for Strong Gravitational Lensing.

Implementation based on He et al. (2024) "Unveiling Lens Light Complexity with A Novel
Multi-Gaussian Expansion Approach for Strong Gravitational Lensing" (arXiv:2403.16253).

The key innovation is organizing Gaussians into "sets" where each set shares center,
axis ratio, and position angle, but can have different orientations between sets.
This enables capturing twisted isophotes common in lens galaxies.
"""

import numpy as np

__all__ = ["MGEMultiSet", "MGEMultiSetPointSource"]


class GaussianEllipse2D:
    """Single 2D elliptical Gaussian profile.

    Surface brightness:
        I(x, y) = I_0 * exp(-R^2 / (2 * sigma^2))

    where R is the elliptical radius.
    """

    def __init__(self):
        pass

    @staticmethod
    def function(x, y, amp, sigma, center_x, center_y, q, phi):
        """Evaluate the 2D elliptical Gaussian.

        Parameters
        ----------
        x, y : array_like
            Coordinates where to evaluate the profile.
        amp : float
            Amplitude (peak intensity).
        sigma : float
            Gaussian width along the major axis.
        center_x, center_y : float
            Center coordinates.
        q : float
            Axis ratio (minor/major), 0 < q <= 1.
        phi : float
            Position angle in radians (counter-clockwise from x-axis).

        Returns
        -------
        array_like
            Surface brightness values.
        """
        # Rotate coordinates
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        dx = x - center_x
        dy = y - center_y
        x_rot = cos_phi * dx + sin_phi * dy
        y_rot = -sin_phi * dx + cos_phi * dy

        # Elliptical radius
        r_ellip_sq = x_rot**2 + (y_rot / q) ** 2

        # Gaussian profile
        return amp * np.exp(-r_ellip_sq / (2 * sigma**2))


class MGESet:
    """A set of Gaussians sharing the same center, axis ratio, and position angle.

    Sigma values are log-spaced between sigma_min and sigma_max.
    Each Gaussian has an independent amplitude (solved linearly).
    """

    def __init__(self, n_gaussians=30):
        """Initialize the MGE set.

        Parameters
        ----------
        n_gaussians : int
            Number of Gaussians in this set.
        """
        self.n_gaussians = n_gaussians
        self._gaussian = GaussianEllipse2D()

    @property
    def num_linear(self):
        """Number of linear parameters (amplitudes)."""
        return self.n_gaussians

    @staticmethod
    def _compute_sigmas(sigma_min, sigma_max, n_gaussians):
        """Compute log-spaced sigma values.

        Parameters
        ----------
        sigma_min : float
            Minimum sigma value.
        sigma_max : float
            Maximum sigma value.
        n_gaussians : int
            Number of Gaussians.

        Returns
        -------
        array
            Log-spaced sigma values.
        """
        if n_gaussians == 1:
            return np.array([np.sqrt(sigma_min * sigma_max)])
        return np.logspace(np.log10(sigma_min), np.log10(sigma_max), n_gaussians)

    def function(self, x, y, amp, sigma_min, sigma_max, center_x, center_y, q, phi):
        """Evaluate the surface brightness of this set.

        Parameters
        ----------
        x, y : array_like
            Coordinates.
        amp : array
            Amplitudes for each Gaussian.
        sigma_min, sigma_max : float
            Range of sigma values.
        center_x, center_y : float
            Center coordinates.
        q : float
            Axis ratio.
        phi : float
            Position angle in radians.

        Returns
        -------
        array_like
            Total surface brightness from all Gaussians in this set.
        """
        sigmas = self._compute_sigmas(sigma_min, sigma_max, self.n_gaussians)
        flux = np.zeros_like(x, dtype=float)

        for i, sigma in enumerate(sigmas):
            flux += self._gaussian.function(
                x, y, amp[i], sigma, center_x, center_y, q, phi
            )
        return flux

    def function_split(self, x, y, amp, sigma_min, sigma_max, center_x, center_y, q, phi):
        """Return individual Gaussian contributions for linear inversion.

        Parameters
        ----------
        x, y : array_like
            Coordinates.
        amp : array
            Amplitudes (typically ones for basis functions).
        sigma_min, sigma_max : float
            Range of sigma values.
        center_x, center_y : float
            Center coordinates.
        q : float
            Axis ratio.
        phi : float
            Position angle in radians.

        Returns
        -------
        list
            List of surface brightness arrays, one per Gaussian.
        """
        sigmas = self._compute_sigmas(sigma_min, sigma_max, self.n_gaussians)
        responses = []

        for i, sigma in enumerate(sigmas):
            response = self._gaussian.function(
                x, y, amp[i], sigma, center_x, center_y, q, phi
            )
            responses.append(response)
        return responses


class MGEMultiSet:
    """Multi-Gaussian Expansion with multiple sets for lens light modeling.

    Each set shares center, axis ratio, and position angle, but different sets
    can have different orientations to capture twisted isophotes.

    All sets share the same center (realistic for galaxies).
    Sigma values are shared across sets.

    Parameters are organized as:
        - sigma_min, sigma_max: Size range (shared)
        - center_x, center_y: Center position (shared)
        - e1_set0, e2_set0, ..., e1_setN, e2_setN: Ellipticity per set
        - amp: Array of all amplitudes (n_sets × n_gaussians_per_set)
    """

    param_names = ["amp", "sigma_min", "sigma_max", "center_x", "center_y"]
    lower_limit_default = {
        "amp": 0,
        "sigma_min": 0.001,
        "sigma_max": 0.01,
        "center_x": -10,
        "center_y": -10,
    }
    upper_limit_default = {
        "amp": 1e10,
        "sigma_min": 1.0,
        "sigma_max": 10.0,
        "center_x": 10,
        "center_y": 10,
    }

    def __init__(self, n_sets=2, n_gaussians_per_set=30):
        """Initialize the multi-set MGE model.

        Parameters
        ----------
        n_sets : int
            Number of Gaussian sets (each with different orientation).
        n_gaussians_per_set : int
            Number of Gaussians per set.
        """
        self.n_sets = n_sets
        self.n_gaussians_per_set = n_gaussians_per_set
        self._sets = [MGESet(n_gaussians_per_set) for _ in range(n_sets)]

        # Build parameter names dynamically
        self.param_names = ["amp", "sigma_min", "sigma_max", "center_x", "center_y"]
        for i in range(n_sets):
            self.param_names.extend([f"e1_set{i}", f"e2_set{i}"])

        # Update limits for ellipticity parameters
        for i in range(n_sets):
            self.lower_limit_default[f"e1_set{i}"] = -0.5
            self.lower_limit_default[f"e2_set{i}"] = -0.5
            self.upper_limit_default[f"e1_set{i}"] = 0.5
            self.upper_limit_default[f"e2_set{i}"] = 0.5

    @property
    def num_linear(self):
        """Total number of linear parameters (amplitudes)."""
        return self.n_sets * self.n_gaussians_per_set

    @staticmethod
    def _e1e2_to_q_phi(e1, e2):
        """Convert ellipticity (e1, e2) to axis ratio (q) and position angle (phi).

        Uses lenstronomy convention:
            e = (1 - q) / (1 + q)
            e1 = e * cos(2*phi)
            e2 = e * sin(2*phi)

        Parameters
        ----------
        e1, e2 : float
            Ellipticity components.

        Returns
        -------
        q : float
            Axis ratio (0 < q <= 1).
        phi : float
            Position angle in radians.
        """
        e = np.sqrt(e1**2 + e2**2)
        e = min(e, 0.9999)  # Prevent q = 0

        if e < 1e-10:
            return 1.0, 0.0

        q = (1 - e) / (1 + e)
        phi = 0.5 * np.arctan2(e2, e1)
        return q, phi

    def function(self, x, y, amp, sigma_min, sigma_max, center_x, center_y, **kwargs):
        """Evaluate total surface brightness.

        Parameters
        ----------
        x, y : array_like
            Coordinates.
        amp : array
            All amplitudes (n_sets × n_gaussians_per_set).
        sigma_min, sigma_max : float
            Size range.
        center_x, center_y : float
            Center coordinates.
        **kwargs : dict
            Must include e1_set0, e2_set0, ..., e1_setN, e2_setN.

        Returns
        -------
        array_like
            Total surface brightness.
        """
        flux = np.zeros_like(x, dtype=float)
        n_g = self.n_gaussians_per_set

        for i, mge_set in enumerate(self._sets):
            e1 = kwargs.get(f"e1_set{i}", 0.0)
            e2 = kwargs.get(f"e2_set{i}", 0.0)
            q, phi = self._e1e2_to_q_phi(e1, e2)

            # Extract amplitudes for this set
            amp_set = amp[i * n_g : (i + 1) * n_g]

            flux += mge_set.function(
                x, y, amp_set, sigma_min, sigma_max, center_x, center_y, q, phi
            )
        return flux

    def function_split(self, x, y, amp, sigma_min, sigma_max, center_x, center_y, **kwargs):
        """Return individual Gaussian contributions for semi-linear inversion.

        Parameters
        ----------
        x, y : array_like
            Coordinates.
        amp : array
            Amplitudes (typically ones for basis functions).
        sigma_min, sigma_max : float
            Size range.
        center_x, center_y : float
            Center coordinates.
        **kwargs : dict
            Must include e1_set0, e2_set0, ..., e1_setN, e2_setN.

        Returns
        -------
        list
            List of surface brightness arrays, one per Gaussian.
        """
        responses = []
        n_g = self.n_gaussians_per_set

        for i, mge_set in enumerate(self._sets):
            e1 = kwargs.get(f"e1_set{i}", 0.0)
            e2 = kwargs.get(f"e2_set{i}", 0.0)
            q, phi = self._e1e2_to_q_phi(e1, e2)

            # Extract amplitudes for this set
            amp_set = amp[i * n_g : (i + 1) * n_g]

            responses.extend(
                mge_set.function_split(
                    x, y, amp_set, sigma_min, sigma_max, center_x, center_y, q, phi
                )
            )
        return responses

    def total_flux(self, amp, sigma_min, sigma_max, center_x, center_y, **kwargs):
        """Compute total integrated flux.

        For a 2D Gaussian with peak I_0 and width sigma, axis ratio q:
            Flux = 2 * pi * I_0 * sigma^2 * q

        Parameters
        ----------
        amp : array
            All amplitudes.
        sigma_min, sigma_max : float
            Size range.
        center_x, center_y : float
            Center coordinates (not used in calculation).
        **kwargs : dict
            Ellipticity parameters for each set.

        Returns
        -------
        float
            Total integrated flux.
        """
        total = 0.0
        n_g = self.n_gaussians_per_set
        sigmas = MGESet._compute_sigmas(sigma_min, sigma_max, n_g)

        for i in range(self.n_sets):
            e1 = kwargs.get(f"e1_set{i}", 0.0)
            e2 = kwargs.get(f"e2_set{i}", 0.0)
            q, _ = self._e1e2_to_q_phi(e1, e2)

            amp_set = amp[i * n_g : (i + 1) * n_g]
            for j, sigma in enumerate(sigmas):
                total += 2 * np.pi * amp_set[j] * sigma**2 * q

        return total

    def light_3d(self, r, amp, sigma_min, sigma_max, center_x, center_y, **kwargs):
        """Compute spherically averaged 3D light density.

        Assumes spherical deprojection (averaging over orientations).

        Parameters
        ----------
        r : array_like
            3D radii.
        amp : array
            Amplitudes.
        sigma_min, sigma_max : float
            Size range.
        center_x, center_y : float
            Center (not used).
        **kwargs : dict
            Ellipticity parameters.

        Returns
        -------
        array_like
            3D light density.
        """
        rho = np.zeros_like(r, dtype=float)
        n_g = self.n_gaussians_per_set
        sigmas = MGESet._compute_sigmas(sigma_min, sigma_max, n_g)

        for i in range(self.n_sets):
            amp_set = amp[i * n_g : (i + 1) * n_g]
            for j, sigma in enumerate(sigmas):
                # Spherical Gaussian deprojection
                rho += amp_set[j] / (np.sqrt(2 * np.pi) * sigma) * np.exp(
                    -(r**2) / (2 * sigma**2)
                )

        return rho


class MGEMultiSetPointSource:
    """MGE model for point source / nuclear emission with independent center.

    This is a single set of Gaussians designed to model compact central emission
    (e.g., AGN, nuclear star clusters) that may have a different center from
    the extended galaxy emission.

    From the paper: "A point set consists of 10 Gaussians, has sigma values
    spanning one-fifth the pixel size to two times the pixel size and has a
    centre that is independent of the centres of the other sets."
    """

    param_names = ["amp", "sigma_min", "sigma_max", "e1", "e2", "center_x", "center_y"]
    lower_limit_default = {
        "amp": 0,
        "sigma_min": 0.001,
        "sigma_max": 0.01,
        "e1": -0.5,
        "e2": -0.5,
        "center_x": -10,
        "center_y": -10,
    }
    upper_limit_default = {
        "amp": 1e10,
        "sigma_min": 0.5,
        "sigma_max": 1.0,
        "e1": 0.5,
        "e2": 0.5,
        "center_x": 10,
        "center_y": 10,
    }

    def __init__(self, n_gaussians=10):
        """Initialize the point source MGE model.

        Parameters
        ----------
        n_gaussians : int
            Number of Gaussians (default 10 from paper).
        """
        self.n_gaussians = n_gaussians
        self._set = MGESet(n_gaussians)

    @property
    def num_linear(self):
        """Number of linear parameters (amplitudes)."""
        return self.n_gaussians

    @staticmethod
    def _e1e2_to_q_phi(e1, e2):
        """Convert ellipticity to axis ratio and position angle."""
        return MGEMultiSet._e1e2_to_q_phi(e1, e2)

    def function(self, x, y, amp, sigma_min, sigma_max, e1, e2, center_x, center_y):
        """Evaluate surface brightness.

        Parameters
        ----------
        x, y : array_like
            Coordinates.
        amp : array
            Amplitudes for each Gaussian.
        sigma_min, sigma_max : float
            Size range.
        e1, e2 : float
            Ellipticity components.
        center_x, center_y : float
            Center coordinates.

        Returns
        -------
        array_like
            Surface brightness.
        """
        q, phi = self._e1e2_to_q_phi(e1, e2)
        return self._set.function(
            x, y, amp, sigma_min, sigma_max, center_x, center_y, q, phi
        )

    def function_split(self, x, y, amp, sigma_min, sigma_max, e1, e2, center_x, center_y):
        """Return individual Gaussian contributions for linear inversion.

        Parameters
        ----------
        x, y : array_like
            Coordinates.
        amp : array
            Amplitudes.
        sigma_min, sigma_max : float
            Size range.
        e1, e2 : float
            Ellipticity components.
        center_x, center_y : float
            Center coordinates.

        Returns
        -------
        list
            List of surface brightness arrays, one per Gaussian.
        """
        q, phi = self._e1e2_to_q_phi(e1, e2)
        return self._set.function_split(
            x, y, amp, sigma_min, sigma_max, center_x, center_y, q, phi
        )

    def total_flux(self, amp, sigma_min, sigma_max, e1, e2, center_x, center_y):
        """Compute total integrated flux.

        Parameters
        ----------
        amp : array
            Amplitudes.
        sigma_min, sigma_max : float
            Size range.
        e1, e2 : float
            Ellipticity components.
        center_x, center_y : float
            Center coordinates (not used).

        Returns
        -------
        float
            Total integrated flux.
        """
        q, _ = self._e1e2_to_q_phi(e1, e2)
        sigmas = MGESet._compute_sigmas(sigma_min, sigma_max, self.n_gaussians)
        total = 0.0
        for j, sigma in enumerate(sigmas):
            total += 2 * np.pi * amp[j] * sigma**2 * q
        return total

    def light_3d(self, r, amp, sigma_min, sigma_max, e1, e2, center_x, center_y):
        """Compute spherically averaged 3D light density.

        Parameters
        ----------
        r : array_like
            3D radii.
        amp : array
            Amplitudes.
        sigma_min, sigma_max : float
            Size range.
        e1, e2 : float
            Ellipticity components (not used in spherical average).
        center_x, center_y : float
            Center (not used).

        Returns
        -------
        array_like
            3D light density.
        """
        rho = np.zeros_like(r, dtype=float)
        sigmas = MGESet._compute_sigmas(sigma_min, sigma_max, self.n_gaussians)

        for j, sigma in enumerate(sigmas):
            rho += amp[j] / (np.sqrt(2 * np.pi) * sigma) * np.exp(
                -(r**2) / (2 * sigma**2)
            )

        return rho


# Helper functions to create standard configurations from the paper
def create_mge_2x30():
    """Create 2×30 MGE model (60 Gaussians)."""
    return MGEMultiSet(n_sets=2, n_gaussians_per_set=30)


def create_mge_4x30():
    """Create 4×30 MGE model (120 Gaussians)."""
    return MGEMultiSet(n_sets=4, n_gaussians_per_set=30)


def create_mge_6x30():
    """Create 6×30 MGE model (180 Gaussians)."""
    return MGEMultiSet(n_sets=6, n_gaussians_per_set=30)
