from itertools import product
import numpy as np
from scipy.linalg import qr, svd, hankel
from scipy.sparse.linalg import svds
from tqdm import tqdm


def generate_data(
        n_timepoints: int,
        n_components: int,
        noise: int,
        n_pixels: int | tuple[int, int],  # n_pixels PER component, if a tuple a random number of pixles within the range is used
        n_pixels_bg,
        interp_factor: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    n_timepoints: int

    n_components: int

    noise: int

    n_pixels: int, or (min, max) range
        n_pixels PER components, if (min, max) range is given then a random number of pixels within this range
        is used for each component

    n_pixels_bg: int

    interp_factor: int
        interpolation factor

    Returns
    -------
    np.ndarray, np.ndarray
        data [pixels, n_timepoints * interp_factor], truth labels: 1D array 0 is background

    """

    clean = np.zeros((n_components, n_timepoints * interp_factor))

    x = 0
    a = 0.7

    b = 0

    firing_prop = 0.05

    spikes = list()

    for i in range(n_components):
        spikes.append((np.random.rand(n_timepoints) < firing_prop).astype(float))

    for c_ix in range(n_components):
        for i in range(2, n_timepoints):
            clean[c_ix, i] = (a * clean[c_ix, i - 1]) + (b * clean[c_ix, i - 2]) + (50 * spikes[c_ix][i])

        x_ = np.arange(0, n_timepoints * interp_factor)
        xp_ = np.linspace(0, n_timepoints * interp_factor, n_timepoints)
        clean[c_ix] = np.interp(x_, xp_, fp=clean[c_ix, :n_timepoints])

    n_timepoints *= 2

    series = list()
    labels = list()

    for i in range(n_components):
        # determine number of pixels for this component
        if isinstance(n_pixels, int):
            _n_pixels_iter = n_pixels
        elif isinstance(n_pixels, tuple):
            _n_pixels_iter = np.random.randint(low=n_pixels[0], high=n_pixels[1])

        for j in range(_n_pixels_iter):
            series.append(clean[i] + np.random.normal(scale=noise, size=n_timepoints))
            labels.append(i + 1)

    # series = np.vstack(series)

    bg_pixels = [np.random.normal(scale=noise, size=n_timepoints) for i in range(n_pixels_bg)]
    series = np.vstack([*series, *bg_pixels])

    return series


class SpaceTimeSVD:
    def __init__(self, movie_vectorized: np.ndarray, w: int):
        self.movie = movie_vectorized
        self.w = w
        self.adjacency = None

    def compute(self, k: int = None):
        print("Computing F, P")
        Ys = list()
        Ps = list()
        Fs = list()

        n_pixels = self.movie.shape[0]
        for i in tqdm(range(n_pixels)):
            y = hankel(self.movie[i, :self.w], self.movie[i, self.w:])
            p = np.flipud(y[:int(self.w / 2)])
            f = y[int(self.w / 2):]

            Ys.append(y)
            Ps.append(p)
            Fs.append(f)

        P_F_shape = (int(self.w / 2) * n_pixels, self.movie.shape[1] - self.w)
        print(self.movie.shape)
        print(P_F_shape)

        P = np.zeros(P_F_shape)
        F = np.zeros(P_F_shape)

        for i in tqdm(range(n_pixels)):
            P[i::n_pixels] = Ps[i]
            F[i::n_pixels] = Fs[i]

        print("computing H")
        H = (F @ P.T) / P.shape[1]

        print("Computing Hp")
        Hp = np.zeros((int(self.w / 2) ** 2, n_pixels ** 2))

        for row_ix, (block_i, block_j) in enumerate(product(*[range(0, int(self.w / 2) * n_pixels, n_pixels)] * 2)):
            Hp[row_ix] = H.T[block_i:block_i + n_pixels, block_j:block_j + n_pixels].ravel()

        del H

        print("computimng svd of Hp")
        if k is None:
            u, s, v = svd(Hp)
        else:
            u, s, v = svds(Hp, k)

        del Hp

        self.adjacency = v[0].reshape(self.movie.shape[0], self.movie.shape[0])


class ClusteringCPQR:
    def __init__(self, adjacency: np.ndarray, k: int):
        self.k = k
        self.B = np.abs(adjacency + adjacency.T) / 2
        self.evals, self.evecs = np.linalg.eigh(self.B)

        self.Qk = self.evecs[:, -self.k:]

        self.Qk.T[:] /= np.linalg.norm(self.Qk.T, axis=0, ord=2)

        _, _, self.pivots = qr(self.Qk.T, pivoting=True)

        self.P = self.Qk.T[:, self.pivots[:self.k]]
        u, s, vT = np.linalg.svd(self.P, full_matrices=False)
        self.W = u @ vT

        self.transformed = self.W.T @ self.Qk.T

        self.labels = self.transformed.argmax(axis=0)
