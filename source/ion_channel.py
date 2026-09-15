from typing import List

import numpy as np
import scipy.stats as st
from scipy.linalg import expm


def getSteadyStateDist(P):
    tolerance = 1e-10

    # need the left eigenvectors
    [u, v] = np.linalg.eig(np.transpose(P))
    v = np.transpose(v)

    index = 0
    for i in u:
        if np.abs(i - 1.) < tolerance:
            return np.real(v[index, :] / np.sum(v[index, :]))
        index += 1

    return None


def randIntNonUniform(p):
    q = np.random.rand()
    r = 0
    for i in range(0, len(p)):
        r += p[i]
        if (q < r):
            return i

    # should never happen
    return -1


class IonChannel:
    def __init__(self, n=1000, C1aExitProb=0.1, TwoConductanceStates=False, open_signal_mag=1.4,
                 closed_signal_mag=0.58):
        self.C1aExitProb = C1aExitProb
        self.TwoConductanceStates = TwoConductanceStates
        self.open_signal_mag = open_signal_mag
        self.closed_signal_mag = closed_signal_mag

        # matrix of rates
        R = np.array(
            [[-9., 9., 0., 0., 0., 0., 0.],  # C1a
             [5., -12.7, 7.7, 0., 0., 0., 0.],  # C1b
             [0., 5.8, -10.7, 4.9, 0., 0., 0.],  # C2
             [0., 0., 10., -17.1, 7.1, 0., 0.],  # O1
             [0., 0., 0., 0., -3., 3., 0.],  # O2
             [0., 0., 0., 0., 7., -13., 6.],  # C3
             [1.7, 0., 0., 0., 0., 12.8, -14.5]])  # C4

        dt = 0.01

        # we specify C1aExitProbability to stay compatible with earlier versions
        # but what it really means is the out rate from C1a normalized by dt
        R[0, 0] = -1 * C1aExitProb / dt
        R[0, 1] = C1aExitProb / dt

        self.P0 = expm(dt * R)

        self.Pmask = np.array(
            [[1., 1., 0., 0., 0., 0., 0.],  # C1a
             [1., 1., 1., 0., 0., 0., 0.],  # C1b
             [0., 1., 1., 1., 0., 0., 0.],  # C2
             [0., 0., 1., 1., 1., 0., 0.],  # O1
             [0., 0., 0., 0., 1., 1., 0.],  # O2
             [0., 0., 0., 0., 1., 1., 1.],  # C3
             [1., 0., 0., 0., 0., 1., 1.]])  # C4)

        if TwoConductanceStates:
            self.statemap = [0., 0., 0., 1., 2., 0., 0.]
        else:
            self.statemap = [0., 0., 0., 1., 1., 0., 0.]

        self.n = n
        self.X, self.y = self.get_receptor_state()

    def get_receptor_state(self):
        r = np.zeros(self.n)
        c = np.zeros(self.n)
        initState = randIntNonUniform(getSteadyStateDist(self.P0))
        for i in range(self.n):
            if i > 0:
                r[i] = randIntNonUniform(self.P0[int(r[i - 1])])
            else:
                r[i] = randIntNonUniform(self.P0[int(initState)])

            c[i] = self.statemap[int(r[i])]

        X = self.get_noisy_receptor_state(c)
        return X, c

    def get_noisy_receptor_state(self, c):
        open_noise = self.open(size=len(c))
        close_noise = self.close(size=len(c))
        noisy_c = np.where(c == 1, open_noise, close_noise)
        return noisy_c

    def open(self, size=1):
        """genhyperbolic noise for open state"""
        params = (1.9865096704542702,
                  0.00199882860659155,
                  -0.0005556910258636614,
                  self.open_signal_mag,
                  0.0002071081765532621)
        genhyperbolic = st.genhyperbolic
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
        start = genhyperbolic.ppf(0.01, *arg, loc=loc, scale=scale) if arg else genhyperbolic.ppf(0.01, loc=loc,
                                                                                                  scale=scale)
        end = genhyperbolic.ppf(0.99, *arg, loc=loc, scale=scale) if arg else genhyperbolic.ppf(0.99, loc=loc,
                                                                                                scale=scale)

        return genhyperbolic.rvs(*params, size=size)

    def close(self, size=1):
        """genhyperbolic noise for closed state"""
        params = (3.6662368635821796,
                  0.5836326667155399,
                  0.16863569591265098,
                  self.closed_signal_mag,
                  0.024246693602449965)
        genhyperbolic = st.genhyperbolic
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
        start = genhyperbolic.ppf(0.01, *arg, loc=loc, scale=scale) if arg else genhyperbolic.ppf(0.01, loc=loc,
                                                                                                  scale=scale)
        end = genhyperbolic.ppf(0.99, *arg, loc=loc, scale=scale) if arg else genhyperbolic.ppf(0.99, loc=loc,
                                                                                                scale=scale)

        return genhyperbolic.rvs(*params, size=size)
