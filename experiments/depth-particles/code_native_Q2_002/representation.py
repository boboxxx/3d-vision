"""Discrete source fitting and quantile-domain law transport; no task inputs."""
import numpy as np


def discrete_W1(p, support, atoms, weights):
    p = np.asarray(p, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    ce = p.cumsum(-1); cs = ce - p
    ae = weights.cumsum(-1); ast = ae - weights
    overlap = np.maximum(np.minimum(ce[..., :, None], ae[..., None, :])
                         - np.maximum(cs[..., :, None], ast[..., None, :]), 0.)
    return (overlap * np.abs(np.asarray(support)[..., :, None]
                            - np.asarray(atoms)[..., None, :])).sum((-2, -1))


def fit(p, z):
    """Every discrete contiguous partition, prefix-sum median L1 objective."""
    c = p.cumsum(-1); c[:, -1] = 1.
    s = (p * z).cumsum(-1)
    cp = np.pad(c, ((0, 0), (1, 0)))
    sp = np.pad(s, ((0, 0), (1, 0)))
    n, bins = p.shape; rows = np.arange(n)
    mass = c.copy(); values = np.empty_like(mass); all_a = np.empty_like(mass); all_b = np.empty_like(mass)
    for k in range(1, bins + 1):
        m = mass[:, k-1]
        ia = (c < (m / 2)[:, None]).sum(-1).clip(0, bins-1)
        ib = (c < ((1+m) / 2)[:, None]).sum(-1).clip(0, bins-1)
        ia = np.where(m == 0, ib, ia); ib = np.where(m == 1, ia, ib)
        a, b = z[ia], z[ib]
        ca, cb = cp[rows, ia], cp[rows, ib]
        sa, sb = sp[rows, ia], sp[rows, ib]
        left = a * (2*ca-m) + s[:, k-1] - 2*sa
        right = b * (2*(cb-m)-(1-m)) + s[:, -1] + s[:, k-1] - 2*sb
        left = np.where(m == 0, 0., left)
        right = np.where(m == 1, 0., right)
        values[:, k-1] = np.maximum(left+right, 0)
        all_a[:, k-1] = a; all_b[:, k-1] = b
    best = (values <= values.min(-1, keepdims=True) + 1e-11).argmax(-1)
    a = all_a[rows, best]; b = all_b[rows, best]; m = mass[rows, best]
    m = np.where(a == b, .5, m)
    return np.stack((a, b), -1), m, best, mass, values


def equal_quantiles(p, z):
    c = p.cumsum(-1); c[:, -1] = 1.
    return np.stack([z[(c < t).sum(-1).clip(0, len(z)-1)] for t in (1/6, 1/2, 5/6)], -1)


def barycentric(atoms, mass):
    a, b = atoms.T
    return np.stack((a, mass*a+(1-mass)*b, b), -1)


def recover_barycentric(depths):
    a, mu, b = depths.T
    mass = np.divide(b-mu, b-a, out=np.full_like(a, .5), where=b>a)
    assert ((mass >= 0) & (mass <= 1)).all()
    return np.stack((a, b), -1), mass
