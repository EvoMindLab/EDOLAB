"""Precision-aware environment storage and the batched NumPy evaluator.

A search evaluates a few points at a time, so the cost of an evaluation is the
number of array calls, not the arithmetic. ``PreparedEnvironment`` therefore
evaluates all components that share a transformation, or a form, in one set
of calls on arrays laid out as (dimension, component, sample): reductions over
the dimension then run along contiguous memory. Parameters are gathered once
per environment into tables already selected by component, and repeated along
the sample axis for the batch sizes in use, so that nearly every operation is
a plain loop over equal shapes.

The formulas are those of the per-point reference in ``problem.py``. Values
can differ from it in the last bits, because sums, products and powers are
taken in a different order.
"""
from __future__ import annotations

from functools import lru_cache
from operator import itemgetter
import threading

import numpy as np

ROWS = ("kappa_plus", "kappa_minus", "p_plus", "p_minus", "rho_plus", "rho_minus")
TRANSFORM_ROWS = {
    "additive_periodic": ("mu_plus", "mu_minus", "gamma_plus", "gamma_minus", "omega_plus", "omega_minus"),
    "log_sinusoidal": ("mu_plus", "mu_minus", "omega1_plus", "omega1_minus", "omega2_plus", "omega2_minus"),
    "wavelet_mod": ("mu_plus", "mu_minus", "omega_plus", "omega_minus"),
    "tensor_interference": ("mu_plus", "mu_minus", "omega_plus", "omega_minus"),
}
# Largest samples * components * dimension evaluated at once; larger batches
# are split, which bounds the temporary arrays.
GROUPED_ITEMS = 65_536
# Most recently used batch sizes whose repeated tables one environment keeps.
REPEATED_SIZES = 8
# Float arrays an evaluation reads. The rotations are used in place instead.
_FORM_ARRAYS = ("c", "beta", "p_single", "rho_single", *ROWS)
_TRANSFORM_ARRAYS = {
    **TRANSFORM_ROWS,
    "log_sinusoidal": (*TRANSFORM_ROWS["log_sinusoidal"], "sigma"),
    "wavelet_mod": (*TRANSFORM_ROWS["wavelet_mod"], "eta"),
    "radial_polytrig": ("mu", "p", "q", "omega"),
}
_RETAINED = 4  # prepared environments kept, most recently built last
_PREPARED: dict[int, "PreparedEnvironment"] = {}
_PREPARING = threading.Lock()
_tobytes = np.ndarray.tobytes


def use_float32(params) -> None:
    """Round a completed environment once, after generating the full trajectory.

    Keeping the generation in its original precision avoids accumulating
    rounding changes in the random walks. The arrays an evaluation reads
    become views of one block of memory: comparing that block with a copy
    detects any later in-place diagnostic edit at the cost of one small copy.
    """
    _round(params)
    owners = [(params, name) for name in _FORM_ARRAYS]
    owners += [(params["T_" + group], name) for group, names in _TRANSFORM_ARRAYS.items() for name in names]
    block = np.empty(sum(owner[name].size for owner, name in owners), dtype=np.float32)
    start = 0
    for owner, name in owners:
        value = owner[name]
        view = block[start:start + value.size].reshape(value.shape)
        view[...] = value
        owner[name] = view
        start += value.size


def _round(params) -> None:
    for name, value in params.items():
        if isinstance(value, np.ndarray) and value.dtype.kind == "f":
            params[name] = value.astype(np.float32)
        elif isinstance(value, dict):
            _round(value)


@lru_cache(maxsize=32)
def other_dimension_indices(dimension: int) -> np.ndarray:
    """For each coordinate, list all other coordinates in their original order."""
    indices = np.broadcast_to(np.arange(dimension), (dimension, dimension))
    return indices[~np.eye(dimension, dtype=bool)].reshape(dimension, dimension - 1)


def tensor_source_key(params):
    """CPU tensors share array storage; rebuild views when an array is replaced."""
    arrays = [params[name] for name in ("c", "R", *ROWS)]
    for name, fields in TRANSFORM_ROWS.items():
        arrays.extend(params["T_" + name][field] for field in fields)
    return (id(params), params.bounds_min, params.bounds_max,
            tuple((id(a), a.shape, a.strides, a.dtype) for a in arrays))


class TorchEnvironment:
    """Only one environment is retained by a problem, bounding tensor-view memory.

    No powers, reductions or transcendentals are moved across components: Torch
    can choose different floating point kernels when operand shapes change.
    CPU arrays stay shared, including in-place diagnostic parameter edits.
    """

    def __init__(self, params, torch, device):
        # Preserve the stored environment's selected evaluation precision.
        dtype = torch.float32 if params.c.dtype == np.float32 else torch.float64

        def tensor(array):
            return torch.as_tensor(array, dtype=dtype, device=device)

        def rows(array):
            t = tensor(array)
            return tuple(t[i:i + 1] for i in range(len(array)))

        self.bounds_min = tensor(params.bounds_min)
        self.bounds_max = tensor(params.bounds_max)
        self.centers = rows(params.c)
        self.rotations = tuple(t.T for t in tensor(params.R).unbind())
        self.rows = {name: rows(params[name]) for name in ROWS}
        self.transforms = {
            name: {field: rows(params["T_" + name][field]) for field in fields}
            for name, fields in TRANSFORM_ROWS.items()
        }


# Tables of one group of G components. A (K, D) parameter becomes (D, G, 1)
# and a per-component value (G, 1); the last axis is the sample axis.

def _rows(arrays, ids, dtype):
    """(K, D) parameters of the components ``ids``, stacked as (len(arrays), D, G, 1)."""
    return np.stack([np.asarray(array)[ids].T for array in arrays]).astype(dtype)[..., None]


def _column(values, dtype):
    return np.asarray(values, dtype=dtype)[:, None]


def _sum(terms):
    """Sum over the dimension, accumulated in double precision and rounded once."""
    return np.add.reduce(terms, axis=0, dtype=np.float64).astype(terms.dtype)


# Each transformation takes the points of its group as a contiguous (D, G, N)
# array. A coordinate selects its parameters by its sign; a zero is unchanged
# by every transformation, so the sign bit decides.

def _additive_periodic_tables(params, ids, dtype):
    group = params["T_additive_periodic"]
    return tuple(_rows((-group["gamma_" + sign], group["omega_" + sign], group["mu_" + sign]), ids, dtype)
                 for sign in ("plus", "minus"))


def _additive_periodic(a, plus, minus):
    decay, omega, mu = np.where(np.signbit(a), minus, plus)
    magnitude = np.abs(a)
    return a + mu * (a * (1 - np.exp(decay * magnitude)) * np.sin(omega * magnitude))


def _log_sinusoidal_tables(params, ids, dtype):
    group = params["T_log_sinusoidal"]
    sigma2 = np.maximum(np.asarray(group["sigma"], dtype=float)[ids] ** 2, np.finfo(float).tiny)
    rate = np.repeat((-1.0 / sigma2).astype(dtype)[None, :, None], params["c"].shape[1], axis=0)
    return (*(_rows((group["omega1_" + sign], group["omega2_" + sign], group["mu_" + sign]), ids, dtype)
              for sign in ("plus", "minus")), rate)


def _log_sinusoidal(a, plus, minus, rate):
    omega1, omega2, mu = np.where(np.signbit(a), minus, plus)
    log_abs = np.log(np.abs(a) + 1e-12)
    # The gate gives T(0)=0 with unit slope there and leaves large |a| unchanged.
    gate = 1.0 - np.exp(a * a * rate)
    modulation = gate * mu * (np.sin(omega1 * log_abs) + np.sin(omega2 * log_abs))
    return np.sign(a) * np.exp(log_abs + modulation)


def _wavelet_mod_tables(params, ids, dtype):
    group = params["T_wavelet_mod"]
    eps = float(group["eps"])
    eta = np.asarray(group["eta"], dtype=float)[ids][:, None]
    tables = []
    for sign in ("plus", "minus"):
        mu = np.asarray(group["mu_" + sign], dtype=float)[ids]
        omega = np.asarray(group["omega_" + sign], dtype=float)[ids]
        # The reference leaves a coordinate unchanged for a vanishing frequency,
        # amplitude or envelope length L = eta / omega (parameter walks reach
        # eta=0, where the envelope tends to zero). A zero amplitude does that.
        unchanged = (omega <= 0) | (mu == 0) | (eta <= 0)
        rate = np.where(unchanged, 1.0, np.maximum(omega, eps) / np.where(eta > 0, eta, 1.0))
        tables.append(_rows((np.where(unchanged, 0.0, mu), omega, rate), slice(None), dtype))
    return (*tables, eps)


def _wavelet_mod(a, plus, minus, eps):
    mu, omega, rate = np.where(np.signbit(a), minus, plus)
    magnitude = np.abs(a)
    squared = np.square(magnitude * rate)
    # sin(omega t - pi/2) = -cos(omega t)
    phi = squared * np.exp(-squared) * np.cos(omega * magnitude)
    return np.where(magnitude < eps, a, a - mu * phi * (a / (magnitude + eps)))


def _tensor_interference_tables(params, ids, dtype):
    group = params["T_tensor_interference"]
    dimension = int(params["d"])
    # The factor keeps the spread of the product gate O(1) in every dimension.
    scale = 2 ** ((dimension - 1) / 2.0)
    tables = []
    for sign in ("plus", "minus"):
        mu = np.asarray(group["mu_" + sign], dtype=float)[ids]
        omega = np.asarray(group["omega_" + sign], dtype=float)[ids]
        # As for the wavelet: a zero weight leaves the coordinate unchanged.
        tables.append(_rows((np.where((mu == 0) | (omega <= 0), 0.0, mu * scale), omega), slice(None), dtype))
    return (*tables, other_dimension_indices(dimension) if dimension <= 8 else None)


def _tensor_interference(a, plus, minus, others):
    weight, omega = np.where(np.signbit(a), minus, plus)
    wave = np.sin(omega * a)
    squares = wave * wave
    # Each coordinate is gated by the product of the other coordinates' squares.
    if others is not None:
        gate = np.multiply.reduce(squares[others], axis=1)
    else:  # products before and after each coordinate: linear in the dimension
        before = np.multiply.accumulate(squares, axis=0)
        after = np.multiply.accumulate(squares[::-1], axis=0)[::-1]
        gate = np.empty_like(squares)
        gate[0] = after[1]
        gate[-1] = before[-2]
        np.multiply(before[:-2], after[2:], out=gate[1:-1])
    return a + weight * wave * gate


def _radial_polytrig_tables(params, ids, dtype):
    group = params["T_radial_polytrig"]
    return (*(_column(np.asarray(group[name])[ids], dtype) for name in ("mu", "omega", "p", "q")),
            float(group["eps"]))


def _radial_polytrig(a, mu, omega, p, q, eps):
    radius = np.sqrt(_sum(a * a))
    phi = radius ** p * np.sin(omega * radius ** q)
    return np.where(radius < eps, a, a + mu * phi / (radius + eps) * a)


# name -> (tables of a group of components, transformation of that group's points)
TRANSFORMS = {
    "additive_periodic": (_additive_periodic_tables, _additive_periodic),
    "log_sinusoidal": (_log_sinusoidal_tables, _log_sinusoidal),
    "wavelet_mod": (_wavelet_mod_tables, _wavelet_mod),
    "tensor_interference": (_tensor_interference_tables, _tensor_interference),
    "radial_polytrig": (_radial_polytrig_tables, _radial_polytrig),
}


def _form_tables(params, order, dtype):
    """Both component forms as one weighted power sum over the coordinates.

    Form A is beta - sum_d rho_d kappa_d |z_d|^(2 p_d). Form B is
    beta - rho (sum_d kappa_d^(1/p) z_d^2)^p: the same sum with exponent two,
    followed by a power and a scale that are one for the components of form A.
    """
    single = np.asarray(params["form"])[order] != 1
    power = np.maximum(np.asarray(params["p_single"], dtype=float)[order], np.finfo(float).eps)
    tables = []
    for sign in ("plus", "minus"):
        kappa = np.asarray(params["kappa_" + sign], dtype=float)[order]
        weight = np.where(single[:, None], kappa ** (1 / power)[:, None],
                          np.asarray(params["rho_" + sign], dtype=float)[order] * kappa)
        exponent = np.where(single[:, None], 2.0, 2 * np.asarray(params["p_" + sign], dtype=float)[order])
        tables.append(_rows((weight, exponent), slice(None), dtype))
    outer = None
    if single.any():
        outer = (_column(np.where(single, np.asarray(params["rho_single"], dtype=float)[order], 1.0), dtype),
                 _column(np.where(single, power, 1.0), dtype))
    return (*tables, _column(np.asarray(params["beta"])[order], dtype), outer)


def _repeated(tables, count):
    """The same tables with every sample axis of length one repeated ``count`` times."""
    if isinstance(tables, np.ndarray):
        return np.repeat(tables, count, axis=-1) if tables.dtype.kind == "f" and tables.shape[-1] == 1 else tables
    if isinstance(tables, (tuple, list)):
        return type(tables)(_repeated(table, count) for table in tables)
    return tables


class PreparedEnvironment:
    """Parameter tables of one environment and its batched evaluation.

    Every evaluation checks that the parameters still have the values read
    here, so replaced arrays and in-place diagnostic edits rebuild the tables.
    """

    def __init__(self, params):
        self.params = params
        centers = params["c"]
        self.dtype = dtype = centers.dtype
        self.count, self.dimension = count, dimension = centers.shape
        self.rotation = params["R"]
        # At most this many samples are evaluated at once.
        self.batch = max(1, GROUPED_ITEMS // (count * dimension))

        # Group components by transformation at each sequence position. Before
        # each position the components are reordered so that every group is a
        # range: a range is written back without index arrays.
        self.sequences = [list(sequence) for sequence in params["transform_seq"]]
        names = [[str(name).lower() for name in sequence] for sequence in self.sequences]
        order = np.arange(count)  # the component at each current position
        steps = []
        for step in range(max(map(len, names), default=0)):
            labels = [names[index][step] if step < len(names[index]) and names[index][step] in TRANSFORMS else ""
                      for index in order]
            present = [name for name in dict.fromkeys(labels) if name]
            rank = {name: position for position, name in enumerate(present)}
            moved = sorted(range(count), key=lambda position: rank.get(labels[position], len(present)))
            order = order[moved]
            groups, start = [], 0
            for name in present:
                stop = start + labels.count(name)
                tables = TRANSFORMS[name][0](params, order[start:stop], dtype)
                groups.append((name, None if stop - start == count else np.arange(start, stop),
                               slice(start, stop), tables))
                start = stop
            steps.append((None if moved == list(range(count)) else np.asarray(moved), groups))
        self.order = None if np.array_equal(order, np.arange(count)) else np.argsort(order)
        # Tables with a sample axis of length one serve any batch size.
        self.tables = (centers[:, :, None], steps, _form_tables(params, order, dtype))
        self.repeated: dict[int, tuple] = {}
        # Points and component values of the latest fitness batch: a passive
        # probe asks for the components of the points a search has just paid for.
        self.latest = None

        used = sorted({name for _, groups in steps for name, *_ in groups})
        self._form_arrays = itemgetter(*_FORM_ARRAYS)
        self._transform_arrays = [("T_" + name, itemgetter(*_TRANSFORM_ARRAYS[name])) for name in used]
        self._eps = ["T_" + name for name in used if name in ("wavelet_mod", "radial_polytrig")]
        self.form = params["form"].tobytes()
        # An environment stored by use_float32 keeps these arrays in one block,
        # and its containers count replaced entries: two cheap comparisons.
        arrays = self._arrays()
        self.containers = [params, *(params[key] for key, _ in self._transform_arrays)]
        self.block = block = arrays[0].base
        if (block is not None and all(array.base is block for array in arrays)
                and all(hasattr(container, "version") for container in self.containers)):
            self.state = [container.version for container in self.containers], block.tobytes()
        else:
            self.block = None
            self.state = self._read(arrays)

    def _arrays(self):
        params = self.params
        arrays = self._form_arrays(params)
        for key, fields in self._transform_arrays:
            arrays += fields(params[key])
        return arrays

    def _read(self, arrays):
        params = self.params
        return [params["d"], *(params[key]["eps"] for key in self._eps), b"".join(map(_tobytes, arrays))]

    def current(self) -> bool:
        params = self.params
        if params["transform_seq"] != self.sequences or params["form"].tobytes() != self.form:
            return False
        if self.block is not None:
            versions, values = self.state
            return [container.version for container in self.containers] == versions and self.block.tobytes() == values
        return params["R"] is self.rotation and self._read(self._arrays()) == self.state

    def _tables(self, count):
        tables = self.repeated.pop(count, None)
        if tables is None:
            if len(self.repeated) >= REPEATED_SIZES:
                del self.repeated[next(iter(self.repeated))]
            tables = _repeated(self.tables, count)
        self.repeated[count] = tables
        return tables

    def values(self, points):
        """Component values (K, N), in the evaluation order, of clipped points given as a (D, N) array."""
        count = points.shape[1]
        centers, steps, (plus, minus, beta, outer) = self._tables(count)
        # z[d, k, n] is coordinate d of R_k (x_n - c_k).
        z = np.empty((self.dimension, self.count, count), dtype=self.dtype)
        np.matmul(self.rotation, points - centers, out=z.transpose(1, 0, 2))
        for moved, groups in steps:
            if moved is not None:
                z = z.take(moved, axis=1)
            for name, ids, positions, tables in groups:
                if ids is None:
                    z = TRANSFORMS[name][1](z, *tables)
                else:
                    z[:, positions] = TRANSFORMS[name][1](z.take(ids, axis=1), *tables)
        weight, exponent = np.where(np.signbit(z), minus, plus)
        penalty = _sum(weight * np.abs(z) ** exponent)
        if outer is not None:
            scale, power = outer
            penalty = scale * penalty ** power
        return beta - penalty

    def _points(self, samples, lower=None, upper=None):
        points = np.array(samples.T, dtype=self.dtype, order="C")
        if lower is not None:
            np.maximum(points, lower, out=points)
            np.minimum(points, upper, out=points)
        return points

    def components(self, samples):
        """Component values (N, K) of (N, D) points that already lie within the bounds."""
        samples = np.asarray(samples)
        if len(samples) <= self.batch:
            points = self._points(samples)
            latest = self.latest
            values = latest[1] if latest is not None and latest[0] == points.tobytes() else self.values(points)
        else:
            values = np.concatenate([self.values(self._points(samples[start:start + self.batch]))
                                     for start in range(0, len(samples), self.batch)], axis=1)
        if self.order is not None:
            values = values.take(self.order, axis=0)
        return np.array(values.T, order="C")

    def fitness(self, samples):
        """Fitness of (N, D) points: the largest component value at their projection on the bounds."""
        params = self.params
        lower, upper = params["bounds_min"], params["bounds_max"]
        if len(samples) <= self.batch:
            points = self._points(samples, lower, upper)
            values = self.values(points)
            self.latest = points.tobytes(), values
            return np.maximum.reduce(values, axis=0).astype(float)
        out = np.empty(len(samples), dtype=float)
        for start in range(0, len(samples), self.batch):
            points = self._points(samples[start:start + self.batch], lower, upper)
            out[start:start + self.batch] = np.maximum.reduce(self.values(points), axis=0)
        return out


def prepare(params) -> PreparedEnvironment:
    """The prepared evaluator of ``params``, rebuilt when its parameters have changed."""
    prepared = _PREPARED.get(id(params))
    if prepared is None or prepared.params is not params or not prepared.current():
        with _PREPARING:
            _PREPARED.pop(id(params), None)
            prepared = _PREPARED[id(params)] = PreparedEnvironment(params)
            while len(_PREPARED) > _RETAINED:
                del _PREPARED[next(iter(_PREPARED))]
    return prepared
