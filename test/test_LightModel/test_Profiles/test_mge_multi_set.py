# -*- coding: utf-8 -*-
"""
Tests for MGE Multi-Set light profiles.

Run with: pytest test_mge_multi_set.py -v
"""

import numpy as np
import numpy.testing as npt
import pytest


class TestMGEMultiSet:
    """Test the MGEMultiSet light profile."""

    def setup_method(self):
        """Setup test fixtures."""
        from lenstronomy.LightModel.Profiles.mge_multi_set import MGEMultiSet
        self.mge = MGEMultiSet(n_sets=2, n_gaussians_per_set=30)

    def test_initialization(self):
        """Test that initialization sets correct parameters."""
        assert self.mge.n_sets == 2
        assert self.mge.n_gaussians_per_set == 30
        assert self.mge.num_linear == 60

    def test_param_names(self):
        """Test that parameter names are generated correctly."""
        expected = [
            "amp", "sigma_min", "sigma_max", "center_x", "center_y",
            "e1_set0", "e2_set0", "e1_set1", "e2_set1"
        ]
        assert self.mge.param_names == expected

    def test_function_output_shape(self):
        """Test that function returns correct shape."""
        x = np.linspace(-2, 2, 50)
        y = np.linspace(-2, 2, 50)
        X, Y = np.meshgrid(x, y)
        x_flat, y_flat = X.flatten(), Y.flatten()

        kwargs = {
            "amp": np.ones(60),
            "sigma_min": 0.1,
            "sigma_max": 2.0,
            "center_x": 0.0,
            "center_y": 0.0,
            "e1_set0": 0.0,
            "e2_set0": 0.0,
            "e1_set1": 0.0,
            "e2_set1": 0.0,
        }

        flux = self.mge.function(x_flat, y_flat, **kwargs)
        assert flux.shape == x_flat.shape

    def test_function_positive(self):
        """Test that output is non-negative for positive amplitudes."""
        x = np.linspace(-2, 2, 20)
        y = np.linspace(-2, 2, 20)
        X, Y = np.meshgrid(x, y)
        x_flat, y_flat = X.flatten(), Y.flatten()

        kwargs = {
            "amp": np.ones(60),
            "sigma_min": 0.1,
            "sigma_max": 2.0,
            "center_x": 0.0,
            "center_y": 0.0,
            "e1_set0": 0.2,
            "e2_set0": 0.1,
            "e1_set1": -0.1,
            "e2_set1": 0.2,
        }

        flux = self.mge.function(x_flat, y_flat, **kwargs)
        assert np.all(flux >= 0)

    def test_function_split_length(self):
        """Test that function_split returns correct number of components."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0])

        kwargs = {
            "amp": np.ones(60),
            "sigma_min": 0.1,
            "sigma_max": 2.0,
            "center_x": 0.0,
            "center_y": 0.0,
            "e1_set0": 0.0,
            "e2_set0": 0.0,
            "e1_set1": 0.0,
            "e2_set1": 0.0,
        }

        responses = self.mge.function_split(x, y, **kwargs)
        assert len(responses) == 60

    def test_function_split_sum(self):
        """Test that sum of split functions equals total function."""
        x = np.linspace(-1, 1, 20)
        y = np.linspace(-1, 1, 20)
        X, Y = np.meshgrid(x, y)
        x_flat, y_flat = X.flatten(), Y.flatten()

        amp = np.random.uniform(0.5, 2.0, 60)
        kwargs = {
            "amp": amp,
            "sigma_min": 0.1,
            "sigma_max": 2.0,
            "center_x": 0.0,
            "center_y": 0.0,
            "e1_set0": 0.1,
            "e2_set0": 0.0,
            "e1_set1": 0.0,
            "e2_set1": 0.1,
        }

        flux_total = self.mge.function(x_flat, y_flat, **kwargs)
        responses = self.mge.function_split(x_flat, y_flat, **kwargs)
        flux_sum = np.sum(responses, axis=0)

        npt.assert_allclose(flux_total, flux_sum, rtol=1e-10)

    def test_centered(self):
        """Test that profile is centered correctly."""
        x = np.array([0.0])
        y = np.array([0.0])

        kwargs = {
            "amp": np.ones(60),
            "sigma_min": 0.1,
            "sigma_max": 2.0,
            "center_x": 0.0,
            "center_y": 0.0,
            "e1_set0": 0.0,
            "e2_set0": 0.0,
            "e1_set1": 0.0,
            "e2_set1": 0.0,
        }

        flux_center = self.mge.function(x, y, **kwargs)

        # Offset center
        kwargs["center_x"] = 1.0
        kwargs["center_y"] = 1.0
        x_offset = np.array([1.0])
        y_offset = np.array([1.0])
        flux_offset = self.mge.function(x_offset, y_offset, **kwargs)

        npt.assert_allclose(flux_center, flux_offset, rtol=1e-10)

    def test_ellipticity_effect(self):
        """Test that ellipticity affects the profile."""
        x = np.array([1.0, 0.0])
        y = np.array([0.0, 1.0])

        # Circular profile
        kwargs_circ = {
            "amp": np.ones(60),
            "sigma_min": 0.1,
            "sigma_max": 2.0,
            "center_x": 0.0,
            "center_y": 0.0,
            "e1_set0": 0.0,
            "e2_set0": 0.0,
            "e1_set1": 0.0,
            "e2_set1": 0.0,
        }
        flux_circ = self.mge.function(x, y, **kwargs_circ)
        
        # Should be symmetric
        npt.assert_allclose(flux_circ[0], flux_circ[1], rtol=1e-10)

        # Elliptical profile (elongated along x)
        kwargs_ell = kwargs_circ.copy()
        kwargs_ell["e1_set0"] = 0.3
        kwargs_ell["e1_set1"] = 0.3
        flux_ell = self.mge.function(x, y, **kwargs_ell)
        
        # Should be brighter along major axis (x)
        assert flux_ell[0] > flux_ell[1]

    def test_total_flux(self):
        """Test total flux calculation."""
        kwargs = {
            "amp": np.ones(60),
            "sigma_min": 0.1,
            "sigma_max": 2.0,
            "center_x": 0.0,
            "center_y": 0.0,
            "e1_set0": 0.0,
            "e2_set0": 0.0,
            "e1_set1": 0.0,
            "e2_set1": 0.0,
        }

        total = self.mge.total_flux(**kwargs)
        assert total > 0

    def test_different_n_sets(self):
        """Test different number of sets."""
        from lenstronomy.LightModel.Profiles.mge_multi_set import MGEMultiSet

        for n_sets in [2, 4, 6]:
            mge = MGEMultiSet(n_sets=n_sets, n_gaussians_per_set=30)
            assert mge.n_sets == n_sets
            assert mge.num_linear == n_sets * 30

            # Check parameter names include all sets
            for s in range(n_sets):
                assert f"e1_set{s}" in mge.param_names
                assert f"e2_set{s}" in mge.param_names


class TestMGEMultiSetPointSource:
    """Test the MGEMultiSetPointSource light profile."""

    def setup_method(self):
        """Setup test fixtures."""
        from lenstronomy.LightModel.Profiles.mge_multi_set import MGEMultiSetPointSource
        self.mge_ps = MGEMultiSetPointSource(n_gaussians=10)

    def test_initialization(self):
        """Test that initialization sets correct parameters."""
        assert self.mge_ps.n_gaussians == 10
        assert self.mge_ps.num_linear == 10

    def test_param_names(self):
        """Test parameter names."""
        expected = ["amp", "sigma_min", "sigma_max", "e1", "e2", "center_x", "center_y"]
        assert self.mge_ps.param_names == expected

    def test_function_output_shape(self):
        """Test that function returns correct shape."""
        x = np.linspace(-1, 1, 20)
        y = np.linspace(-1, 1, 20)
        X, Y = np.meshgrid(x, y)
        x_flat, y_flat = X.flatten(), Y.flatten()

        kwargs = {
            "amp": np.ones(10),
            "sigma_min": 0.01,
            "sigma_max": 0.1,
            "e1": 0.0,
            "e2": 0.0,
            "center_x": 0.0,
            "center_y": 0.0,
        }

        flux = self.mge_ps.function(x_flat, y_flat, **kwargs)
        assert flux.shape == x_flat.shape

    def test_compact(self):
        """Test that point source is compact."""
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([0.0, 0.0, 0.0])

        kwargs = {
            "amp": np.ones(10),
            "sigma_min": 0.01,
            "sigma_max": 0.1,
            "e1": 0.0,
            "e2": 0.0,
            "center_x": 0.0,
            "center_y": 0.0,
        }

        flux = self.mge_ps.function(x, y, **kwargs)
        
        # Should decrease with distance
        assert flux[0] > flux[1] > flux[2]

    def test_function_split_length(self):
        """Test that function_split returns correct number of components."""
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])

        kwargs = {
            "amp": np.ones(10),
            "sigma_min": 0.01,
            "sigma_max": 0.1,
            "e1": 0.0,
            "e2": 0.0,
            "center_x": 0.0,
            "center_y": 0.0,
        }

        responses = self.mge_ps.function_split(x, y, **kwargs)
        assert len(responses) == 10


class TestHelperFunctions:
    """Test helper functions."""

    def test_create_mge_2x30(self):
        """Test 2x30 factory function."""
        from lenstronomy.LightModel.Profiles.mge_multi_set import create_mge_2x30
        mge = create_mge_2x30()
        assert mge.n_sets == 2
        assert mge.n_gaussians_per_set == 30

    def test_create_mge_4x30(self):
        """Test 4x30 factory function."""
        from lenstronomy.LightModel.Profiles.mge_multi_set import create_mge_4x30
        mge = create_mge_4x30()
        assert mge.n_sets == 4
        assert mge.n_gaussians_per_set == 30

    def test_create_mge_6x30(self):
        """Test 6x30 factory function."""
        from lenstronomy.LightModel.Profiles.mge_multi_set import create_mge_6x30
        mge = create_mge_6x30()
        assert mge.n_sets == 6
        assert mge.n_gaussians_per_set == 30


class TestLinearAmplitudeRecovery:
    """Test that linear amplitudes can be recovered via inversion."""

    def test_linear_recovery(self):
        """Test that amplitudes can be recovered from data."""
        from lenstronomy.LightModel.Profiles.mge_multi_set import MGEMultiSet
        from scipy.optimize import nnls

        mge = MGEMultiSet(n_sets=2, n_gaussians_per_set=10)

        # Create coordinates
        x = np.linspace(-2, 2, 30)
        y = np.linspace(-2, 2, 30)
        X, Y = np.meshgrid(x, y)
        x_flat, y_flat = X.flatten(), Y.flatten()

        # True amplitudes
        np.random.seed(42)
        amp_true = np.random.uniform(0.5, 2.0, 20)

        # Non-linear parameters
        kwargs = {
            "amp": amp_true,
            "sigma_min": 0.2,
            "sigma_max": 1.5,
            "center_x": 0.0,
            "center_y": 0.0,
            "e1_set0": 0.1,
            "e2_set0": 0.0,
            "e1_set1": 0.0,
            "e2_set1": 0.1,
        }

        # Generate "data"
        data = mge.function(x_flat, y_flat, **kwargs)

        # Get basis functions
        kwargs_unit = kwargs.copy()
        kwargs_unit["amp"] = np.ones(20)
        basis = mge.function_split(x_flat, y_flat, **kwargs_unit)

        # Build design matrix
        A = np.array(basis).T

        # Solve for amplitudes
        amp_solved, _ = nnls(A, data)

        # Check recovery
        npt.assert_allclose(amp_true, amp_solved, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
