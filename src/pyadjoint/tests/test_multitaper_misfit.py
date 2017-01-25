#!/usr/bin/env pythonG
# -*- coding: utf-8 -*-
"""
Automated tests for multitaper_misfit.py and make sure it will work
and do something expected.

:copyright:
    Youyi Ruan (youyir@princeton.edu), 2017
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

import pyadjoint


@pytest.fixture(params=list(pyadjoint.AdjointSource._ad_srcs.keys()))
def adj_src(request):
    """
    Fixture returning the name of all adjoint sources.
    """
    return request.param


def compare_error(ref, syn):
    """
    calculate difference between two adjoint source in ascii file
    """

    corr_min = 1.0
    err_max = 0.0

    # correlation test
    corr_mat = np.corrcoef(ref, syn)
    corr = np.min(corr_mat)
    corr_min = min(corr, corr_min)

    # least square test
    norm = np.linalg.norm
    sqrt = np.sqrt
    err = norm(ref-syn)/sqrt(norm(ref)*norm(syn))
    err_max = max(err, err_max)

    return corr_min, err_max


# def test_multitaper_adjoint_source(adj_src):
def test_multitaper_adjoint_source():
    """
    Make sure multitaper adjoint source is working correctly .
    """

    src_type = "multitaper_misfit"

    obs, syn = pyadjoint.utils.get_example_sac_data()
    measurement = pyadjoint.utils.get_example_mt_measurement()
    adjsrc_dt_ref, adjsrc_am_ref = pyadjoint.utils.get_example_mt_adjsrc()

    window = [[3313.6, 3756.0]]

    tol_corr = 0.99
    tol_err = 3.E-02
#    rtol = 1.0E-05

    config = pyadjoint.ConfigMultiTaper(min_period=60.0,
                                        max_period=100.0,
                                        lnpt=15,
                                        transfunc_waterlevel=1.0E-10,
                                        water_threshold=0.02,
                                        ipower_costaper=10,
                                        min_cycle_in_window=0.5,
                                        taper_percentage=1.0,
                                        taper_type='cos_p10',
                                        mt_nw=4,
                                        num_taper=8,
                                        dt_fac=2.0,
                                        phase_step=1.5,
                                        err_fac=2.5,
                                        dt_max_scale=3.5,
                                        measure_type='dt',
                                        dt_sigma_min=1.0,
                                        dlna_sigma_min=0.5,
                                        use_cc_error=True,
                                        use_mt_error=False)

    for catlog in ("dt", "am"):

        config.measure_type = catlog

        a_src = pyadjoint.calculate_adjoint_source(adj_src_type=src_type,
                                                   observed=obs,
                                                   synthetic=syn,
                                                   config=config,
                                                   window=window,
                                                   adjoint_src=True,
                                                   plot=False)

        assert isinstance(a_src.adjoint_source, np.ndarray)

        print(len(a_src.adjoint_source), len(adjsrc_dt_ref))
        if catlog == "dt":
            corr_min, err_max = compare_error(a_src.adjoint_source,
                                              adjsrc_dt_ref[::-1])
            measurement_ref = measurement["IU.KBL..BHZ.mt.dt.adj"]
            measurement_new = a_src.measurement[0]

        if catlog == "am":
            corr_min, err_max = compare_error(a_src.adjoint_source,
                                              adjsrc_am_ref[::-1])
            measurement_ref = measurement["IU.KBL..BHZ.mt.am.adj"]
            measurement_new = a_src.measurement[0]

        print("corr_min, err_max", corr_min, err_max)

        assert corr_min >= tol_corr
        assert err_max <= tol_err

        np.testing.assert_allclose(measurement_ref["dt"],
                                   measurement_new["dt"])

        np.testing.assert_allclose(measurement_ref["misfit_dt"],
                                   measurement_new["misfit_dt"])

        np.testing.assert_allclose(measurement_ref["dlna"],
                                   measurement_new["dlna"])

        np.testing.assert_allclose(measurement_ref["misfit_dlna"],
                                   measurement_new["misfit_dlna"])

        assert measurement_ref["type"] == measurement_new["type"]
