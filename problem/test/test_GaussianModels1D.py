from problem.GaussianModels1D import *

import numpy as np
from numpy.testing import assert_equal, assert_allclose


def test_psi():
    np.random.seed(42)
    x = np.random.uniform(size=10)
    x0 = np.random.randint(100, size=10) * 1e-5
    a = np.random.uniform(size=10)

    answer = []
    for x_, x0_, a_ in zip(x, x0, a):
        answer.append(psi(x_, x0_, a_))

    true_answer = [0.7261885406582674,
                   0.4547370403979735,
                   0.2580024520624825,
                   0.345236803487602,
                   0.7506477464565775,
                   0.7034633527886424,
                   0.4150983826153345,
                   0.42781989460424946,
                   0.5704870884260796,
                   0.46831947115911204]

    assert_allclose(answer, true_answer)


def test_normalization():
    np.random.seed(42)
    x0 = np.random.randint(100, size=10) * 1e-5
    a = np.random.uniform(size=10)

    answer = []
    for x0_, a_ in zip(x0, a):
        answer.append(normalization(x0_, a_))
    true_answer = np.ones(10)
    assert_allclose(answer, true_answer)


def test_compute_energy_0():
    np.random.seed(42)
    D = np.random.randint(100, size=10) * 1e-5
    a = np.random.uniform(size=10)

    answer = []
    for D_, a_ in zip(D, a):
        answer.append(compute_energy_0(D_, a_))
    true_answer = [0.009998922808085718,
                   0.010838250409858518,
                   0.004838206526738479,
                   0.0012014778935433529,
                   0.011461777556575115,
                   0.006251521684069666,
                   0.005777690329884134,
                   0.005392388979384332,
                   0.005084932130324375,
                   0.0067623206374770786]
    assert_allclose(answer, true_answer)


def test_compute_energy_1():
    np.random.seed(42)
    D = np.random.randint(100, size=10) * 1e-5
    a = np.random.uniform(size=10)

    answer = []
    for D_, a_ in zip(D, a):
        answer.append(compute_energy_1(D_, a_))
    true_answer = [-0.00482295875104286,
                   -0.005137635074375239,
                   -0.002365589005361361,
                   -0.0005267327752520315,
                   -0.005524358600263428,
                   -0.0030504700545318825,
                   -0.002677188828033714,
                   -0.002484350309337573,
                   -0.0023533143129860386,
                   -0.003171588697442606]
    assert_allclose(answer, true_answer)


def test_compute_expected_energy():
    np.random.seed(42)
    alpha = np.random.randint(0, 10)
    D = np.random.randint(100, size=10) * 1e-2
    a = np.random.uniform(size=10)

    answer = []
    for D_, a_ in zip(D, a):
        answer.append(compute_expected_energy(alpha, D_, a_))
    true_answer = [2.5074661846978614,
                   2.1022116559985045,
                   2.8639561808173735,
                   2.2906081756312653,
                   2.4228703394802538,
                   2.806595776944829,
                   2.1871600362729002,
                   2.1464422759915163,
                   2.2655909653779456,
                   2.269206112217431]
    assert_allclose(answer, true_answer)


def test_compute_MOs():
    np.random.seed(42)
    n_sites = 5
    alpha = float(np.random.randint(1, 10))
    D = np.random.randint(100, size=10) * 1e-2
    a = np.random.uniform(size=10)
    R = float(np.random.randint(1, 6))

    answer_e = []
    for D_, a_ in zip(D, a):
        e, mo = compute_MOs(n_sites,
                            alpha,
                            D_,
                            a_,
                            R)
        answer_e.append(e)

    answer_e = np.array(answer_e)
    true_answer_e = np.array([[3.00508051, 3.00508051, 3.00508051, 3.00508051, 3.00508051],
                              [2.59913583, 2.59913583, 2.59913583, 2.59913633, 2.59913633],
                              [3.36341148, 3.36341148, 3.36341148, 3.36341148, 3.36341148],
                              [1.76004323, 1.84392604, 1.84392604, 2.26813414, 2.26813414],
                              [2.91976266, 2.91976266, 2.91976266, 2.91976266, 2.91976266],
                              [3.30569334, 3.30569334, 3.30569334, 3.30569334, 3.30569334],
                              [2.67742743, 2.67742743, 2.67742743, 2.68178804, 2.68178804],
                              [2.62630441, 2.62630442, 2.62630442, 2.63591761, 2.63591761],
                              [2.74888173, 2.74888174, 2.74888174, 2.75684087, 2.75684087],
                              [2.76705575, 2.76705575, 2.76705575, 2.76748386, 2.76748386]])
    assert_allclose(answer_e, true_answer_e)


def test_compute_total_energy():
    np.random.seed(42)
    n_sites = 5
    epsilon = np.random.rand(10, 10) * 1e-2

    answer = []
    for epsilon_ in epsilon:
        answer.append(compute_total_energy(n_sites, epsilon_))
    true_answer = [0.03382502792325963,
                   0.028134313337160155,
                   0.01794838159284061,
                   0.016211895441627395,
                   0.012688188110273165,
                   0.04428933843815536,
                   0.021487901520786855,
                   0.01947443018785262,
                   0.03303701130258952,
                   0.024264631149394908]
    assert_allclose(answer, true_answer)


def test_compute_Peierls_MOs():
    np.random.seed(42)
    n_sites = 5
    alpha = float(np.random.randint(1, 10))
    D = np.random.randint(100, size=10) * 1e-2
    a = np.random.uniform(size=10)
    R = float(np.random.randint(1, 6))
    xi = float(np.random.randint(1, 10))

    answer_e = []
    for D_, a_ in zip(D, a):
        epsilon, orbs = compute_Peierls_MOs(n_sites,
                                            alpha,
                                            D_,
                                            a_,
                                            R,
                                            xi)
        answer_e.append(epsilon)

    answer_e = np.array(answer_e)

    true_answer_e = np.array([[3.00475972, 3.00475972, 3.00508051, 3.00508051, 3.00508051],
                              [2.59410368, 2.59410368, 2.59913683, 2.59913683, 2.59913683],
                              [3.3630945, 3.3630945, 3.36341149, 3.36341149, 3.36341149],
                              [2.08783694, 2.08801417, 2.58683138, 2.67767169, 2.69947696],
                              [2.91959739, 2.91959739, 2.91976266, 2.91976266, 2.91976266],
                              [3.30552839, 3.30552839, 3.30569335, 3.30569335, 3.30569335],
                              [2.56227306, 2.56227306, 2.68614865, 2.68614865, 2.68614865],
                              [2.47566031, 2.47566031, 2.64553078, 2.64553079, 2.64553079],
                              [2.62065117, 2.62065117, 2.76479999, 2.7648, 2.7648],
                              [2.71771473, 2.71771473, 2.76791196, 2.76791196, 2.76791196]])
    assert_allclose(answer_e, true_answer_e)
