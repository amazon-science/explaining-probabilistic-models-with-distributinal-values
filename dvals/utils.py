import copy
from itertools import chain, combinations
import torch

import numpy as np
from functional import seq


def superset(A, B):
    """
    Returns True if A is a superset of B
    """
    if isinstance(A, set) and isinstance(B, set):
        return B.issubset(A)
    else:
        raise NotImplementedError()


def powerset(iterable):
    s = list(iterable)
    return map(set, chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))


def get_rng():
    return np.random.default_rng()

class DiscreteRV:

    dict_cls = dict

    def __init__(self, *args, **kwargs):
        self._dict = self.dict_cls(*args, **kwargs)
        self._hash = None

    def validate(self):
        _sum = np.sum(self.values(as_array=True))
        if not np.allclose(_sum, 1.):
            raise ValueError(f"Probabilities do not sum to 1., found {_sum}")
        return self

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self.keys()

    def __len__(self):
        return len(self.keys())

    def __repr__(self):
        return '<%s %r>' % (self.__class__.__name__, self._dict)

    def __hash__(self):
        if self._hash is None:
            h = 0
            for key, value in self.items():
                h ^= hash((key, value))
            self._hash = h
        return self._hash

    def norm(self):
        return np.sum(self.values(as_array=True, normalized=False))

    def normalized(self):
        nrm = self.norm()
        cp = copy.deepcopy(self)
        for k, v in self.items():
            cp._dict[k] = v/nrm
        return cp

    def items(self, normalized=False):
        return zip(self.keys(), self.values(normalized=normalized))

    def keys(self, as_array=False):
        keys = self._dict.keys()
        return np.array(tuple(keys)) if as_array else keys

    @property
    def support(self):
        """
        :return: The support of this discrete random variable as a numpy array, each row is a point.
        Hence, len(support) is the cardinality of this RV (i.e. on how many point it is non-zero) and len(support[0])
        is the dimensionality of each element in the support (if the random variable is embedded in R^d for some d)
        """
        return self.keys(as_array=True)

    def values(self, as_array=False, normalized=False):
        nrm = self.norm() if normalized else 1
        values = map(lambda v: v/nrm, self._dict.values())
        return np.array(tuple(values)) if as_array else values

    @property
    def probabilities(self):
        return self.values(as_array=True, normalized=True)

    def __ior__(self, other):
        if self._hash:
            raise RuntimeError("The hash code has already been generated!")
        if isinstance(other, tuple):
            # realisation, probability
            # here we take it that you're simply building up your discrete RV,
            # so that we do not check for validity, etc.
            self._insert(other[0], other[1])
        else:
            self._insert(other)
        return self

    def _insert(self, value, probability=None):
        if isinstance(value, self.__class__):
            # take this to be the update of the PMF (e.g. towards computing
            # the q_i = E_S q_{i, S}
            for val, prob in value.items():
                self._insert(val, probability*prob if probability else prob)
        else:
            if probability is None:
                # interpret it as empirical random variable
                if value in self:  # update counter
                    self._dict[value] += 1
                else:
                    self._dict[value] = 1
            else:
                if value in self:
                    self._dict[value] += probability
                else:
                    self._dict[value] = probability

    def top_k(self, k=1):
        tpk = seq(self.items())\
            .sorted(lambda k_v: -k_v[1])\
            .take(k).dict()
        return tpk

    def entropy(self, eps=1.e-5):
        prob_dist = self.values(True, True)
        return -np.sum(prob_dist[prob_dist > eps] * np.log2(prob_dist[prob_dist > eps]))

    def expected_value(self):
        return np.sum(self.probabilities.reshape((-1, 1)) * self.support, axis=0)

    def variance(self):
        d = len(self.support[0])
        mu = self.expected_value()
        cov = np.zeros((d, d))
        for p, s in zip(self.probabilities, self.support):
            cov += p * (s - mu).reshape((-1, 1)) @ (s - mu).reshape((1, -1))
        return cov


def reduce_classes(mod, full_out, n_to_keep):
    """Given a torch multiclass classifier that returns the logit
    limits the number of classes to `n_to_keep` (using the top-`n-to-keep` classes according to `full_out`)
    and adds a class that represents the union of the remaining classes.


    :return: the kept indices and a new callable representing the reduced model"""
    _, in_indices = torch.topk(full_out, n_to_keep)
    out_indices = seq(range(len(full_out))) \
        .filter(lambda i: i not in in_indices) \
        .list()

    def _process(x):
        logits = mod(x)
        in_logits = logits[:, in_indices]
        out_logits = logits[:, out_indices]
        out_logits = torch.logsumexp(out_logits, dim=1)
        return torch.cat([in_logits, out_logits[:, None]], dim=1)

    return in_indices, _process


def maybe_convert(obj):
    if not isinstance(obj, (list, tuple)):
        return tuple(obj)
    return obj


def maybe_call(fn, arg, return_arg=False):
    if callable(fn):
        return fn(arg)
    return arg if return_arg else fn
