"""
This module contains basic definitions for games (as set functions) and some special class of games
"""
from typing import Callable

import numpy as np

from .categorical_diffs import joint_probability
from .utils import DiscreteRV, maybe_convert, maybe_call, superset, powerset, get_rng

import itertools
import numbers
from abc import ABC

from functional import seq


class Game:

    def __init__(self, val_function=None):  # , d: int=None) -> None:
        """
        This represents an abstract cooperative game.
        """
        super().__init__()
        self._val_function = val_function

    def __call__(self, S, *args, **kwargs):
        """
        Returns the payoff of the game played by the coalition `S`.
        Batching should be allowed (currently in the form of numpy vectors of sets,
        or any other iterable).
        """
        if self._val_function is None:
            raise ValueError()
        if isinstance(S, set): S = np.array([S])
        return self._val_function(S, *args, **kwargs)

    def __rmul__(self, c):
        """
        Left multiplication by a scalr

        :param c: a scalr
        :return: a game with payoff v(S) = a * self(S)
        """
        return Game(lambda S: c * self(S))

    def __radd__(self, other):
        """
        Sum of two games (assuming same set of players/roles)
        :param other: another game
        :return: a game with payoff v(S) = self(S) + other(S)
        """
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __add__(self, other):
        if isinstance(other, Game):
            return Game(lambda S: self(S) + other(S))
        raise ValueError()


class CarrierGame(Game):

    def __init__(self, T):  # d: int = None, ) -> None:
        """
        A carrier game with carrier set `T` is a game that returns 1 iif
        S superset T. Carrier games are simple monotonic superadditive games
        and may provide a base for the vector space of (general)
        cooperative games.
        """
        super().__init__()  # d)
        self.T = T
        # defer to backend?
        self._val_function = lambda S: np.vectorize(
            lambda s: superset(s, self.T)
        )(np.atleast_1d(np.asarray(S))).astype(float)


class ElementaryGame(Game):

    def __init__(self, T):  # d: int = None, ) -> None:
        """
        An "elementary" game on the set `T` is a game that returns 1 iif
        S = T. Elementary games are simple but not monotonic
        and may provide a (canonical) base for the vector space of (general)
        cooperative games.
        """
        super().__init__()  # d)
        self.T = T
        # defer to backend?
        self._val_function = lambda S: np.vectorize(
            lambda s: self.T == s
        )(np.atleast_1d(np.asarray(S))).astype(float)


def base(players, base_game=ElementaryGame):
    """
    Returns a base using the class defined by `base_game`

    :param players: (int or iterable) elements for the base
    :param base_game: class for the base
    :return: a list of games that form a base for the elements
    """
    assert base_game == ElementaryGame or base_game == CarrierGame, f"can't do a base with {base_game}"
    if isinstance(players, int):
        players = list(range(players))
    return [base_game(T) for T in powerset(players)]


def random_game(players, sparsity=0.05, payoffs_dist=None, rng=None, base_game=ElementaryGame):
    """
    Creates a random game with given players. Can specify the sparsity rate, the coefficient distribution and
    the base class.

    :param players: int or list of players, if int will consider the players as numbered from q to max
    :param sparsity:
    :param payoffs_dist:
    :param rng:
    :param base_game:

    :return: a `Game` object
    """
    _base = base(players, base_game=base_game)

    if isinstance(players, int):
        players = list(range(players))

    d = len(players)
    if not rng:
        rng = get_rng()
    if not payoffs_dist:
        payoffs_dist = rng.normal
    sparse_coefficients = np.zeros((2 ** d,))
    lst = list(range(2 ** d))
    rng.shuffle(lst)
    sparse_coefficients[lst[:int(sparsity * 2 ** d)]] = 1.

    coefficients = payoffs_dist(size=(2 ** d)) * sparse_coefficients

    game = sum([c*e for c, e in zip(coefficients, _base) if c != 0.])
    game._coefficients = coefficients  # just save coefficients for debugging
    game._base = _base
    return game

# ---------------------------------
# Stochastic games
# ---------------------------------



class Payoff(ABC):

    def __rsub__(self, other):
        if other == 0:
            return self
        else: return self - other

    def __sub__(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError(f"other must be an instance of {self.__class__}, found {other.__class__}")
        return self.diff(other)

    def diff(self, other): ...

    def __eq__(self, other): ...

    def __hash__(self):
        return hash(self)

    def empty_diff(self): ...


class Difference(ABC):

    @classmethod
    def zero(cls): ...

    def __rmul__(self, other):
        if other == 0.:
            return self.zero()
        elif other == 1.:
            return self
        elif isinstance(other, numbers.Number):
            return self.scalar_mult(other)
        else:
            raise ValueError(f"other must be a scal, found {other.__class__}")

    def scalar_mult(self, other): ...

    def __radd__(self, other):
        """
        Sum of two games (assuming same set of players/roles)
        :param other: another game
        :return: a game with payoff v(S) = self(S) + other(S)
        """
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __add__(self, other):
        if other == self.zero():
            return self
        elif isinstance(other, self.__class__):
            return self.sum(other)
        raise ValueError(f"other must be an instance of {self.__class__}, found {other.__class__}")

    def sum(self, other): ...


class _DiscreteRVPayoff(Payoff, ABC, DiscreteRV):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class _DiscreteRVDifference(Difference, DiscreteRV, ABC):
    ZERO_POINT = 0.  # representation of the zero Dirac delta

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    _zero = DiscreteRV([(0., 1.)])

    @classmethod
    def zero(cls):
        return _DiscreteRVDifference._zero

    def probability_of_change(self):
        """
        :return: The probability that this RV is non-zero (indicating the probability
                    of any change in the outcome).
        """
        return 1. - self[self.ZERO_POINT]/self.norm()

    def top_k(self, k=1, normalized=False):
        """
        Overrides  `DiscreteRV.top_k` to exclude the probability of no change.
        It also returns the remaining mass,
        """
        tpk = seq(self.items())\
            .sorted(lambda k_v: -k_v[1])\
            .filter(lambda k_v: k_v[0] != self.ZERO_POINT)\
            .take(k)
        if normalized:
            nrm = self.norm()
            tpk = tpk.map(lambda k_v: (k_v[0], k_v[1]/nrm))
            div = 1
        else: div = self.norm()
        tpk = tpk.dict()
        return {
            'top-k': tpk,
            'remaining mass': self.probability_of_change()
                              - seq(tpk.values()).sum()/div
        }


# --- BERNOULLI ----

class BernoulliPayoff(_DiscreteRVPayoff):

    def __init__(self, pi):
        """
        :param pi: success probability, real value in [0, 1]
        """
        self._pi = pi
        super().__init__(((1., pi), (0, 1 - pi)))

    # noinspection PyProtectedMember
    def diff(self, other):
        pi1, pi2 = self._pi, other._pi
        _min, _max = min(pi1, pi2), max(pi1, pi2)
        qpm, qmm = pi1 - _min, pi2 - _min
        q0 = 1. - qpm - qmm
        return BernoulliDifference({
            1.: qpm,
            0.: q0,
            -1.: qmm})

    def empty_diff(self):
        return BernoulliDifference()

    @property
    def support(self):
        return super().support.reshape((-1, 1))


class BernoulliDifference(_DiscreteRVDifference):

    def scalar_mult(self, other):
        raise NotImplementedError

    def sum(self, other):
        raise NotImplementedError

    @property
    def support(self):
        return super().support.reshape((-1, 1))


class StochasticGame(Game):
    pass

class BernoulliGame(StochasticGame):

    @staticmethod
    def from_success_prob_game(v):
        def _payoff(S):
            _lst = tuple(BernoulliPayoff(pi) for pi in v(S))
            ary = np.array(_lst, dtype=object)
            return ary

        return BernoulliGame(_payoff)


# --- CATEGORICAL --------

def softmax(x):
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()


class CategoricalPayoff(_DiscreteRVPayoff):

    _diff_elements_store = {}
    _diff_elements_dict_store = {}

    def __init__(self, logits, categories=None) -> None:
        self._logits = logits
        self._categories = categories or tuple(range(np.shape(logits)[0]))
        self._n_cat = len(self._categories)
        self._probs = softmax(logits)
        super().__init__(zip(self._categories, self._probs))

    @property
    def support(self):
        return np.eye(self._n_cat)

    @property
    def categories(self):
        return self._categories

    def _diff_elements(self):
        if self._categories in CategoricalPayoff._diff_elements_store:
            return CategoricalPayoff._diff_elements_store[self._categories], \
                CategoricalPayoff._diff_elements_dict_store[self._categories]
        else:
            CategoricalPayoff._diff_elements_store[self._categories] = tuple(
                filter(lambda a: a[0] != a[1],
                       itertools.product(self._categories, repeat=2))
            ) + (CategoricalDifference.ZERO_POINT, )
            CategoricalPayoff._diff_elements_dict_store[self._categories] = \
                {e: k for k, e in enumerate(self._categories)} if \
                    not isinstance(self._categories[0], int) else None

            return self._diff_elements()

    def diff(self, other):
        # noinspection PyProtectedMember
        th1, th2 = self._logits, other._logits
        result = joint_probability(th1, th2)
        return CategoricalDifference(result, *self._diff_elements())

    def empty_diff(self):
        _res = {
            "joint": np.zeros((1, len(self), len(self)))
        }
        ct = CategoricalDifference(_res, *self._diff_elements())
        return ct


class CategoricalDifference(_DiscreteRVDifference):

    def __init__(self, res, elements, element_dict=None):
        super().__init__()
        self._elements = elements  # these represent the pairs (to, from) of the transition polytope
        self._res = res  # for future use
        self._joint_pmf = res["joint"][0]
        self._element_dict = element_dict
        self._zero_val = None
        self._n_cat = len(self._joint_pmf)
        # does not initialise the dict view

    def __getitem__(self, key):
        if isinstance(key, np.ndarray):
            # ew representation with arrays as centers
            key = (np.argmax(key).item(), np.argmin(key).item())
        if key in self._elements:
            if key == self.ZERO_POINT:
                if self._zero_val is None:
                    self._zero_val = np.sum(np.diag(self._joint_pmf))
                return self._zero_val
            else:
                c1, c2 = key
                if not isinstance(c1, int) or not isinstance(c2, int):
                    c1, c2 = self._element_dict[c1], self._element_dict[c2]

                return self._joint_pmf[c1, c2]
        else:
            print(f"{key} not found!")
            raise KeyError()

    def keys(self, as_array=False):
        keys = self._elements
        return np.array(tuple(keys)) if as_array else keys

    def values(self, as_array=False, normalized=False):
        nrm = self.norm() if normalized else 1
        values = map(lambda v: v / nrm, (self[c] for c in self._elements))
        return np.array(tuple(values)) if as_array else values

    def __repr__(self):
        return '<%s %r>' % (self.__class__.__name__, dict(self.items()))

    @property
    def support(self):
        E = np.eye(self._n_cat)
        support = np.zeros((self._n_cat*(self._n_cat - 1) + 1, self._n_cat))
        for k, (i, j) in enumerate(seq(itertools.product(range(self._n_cat), repeat=2)).filter(lambda a: a[0] != a[1])):
            support[k] = E[i] - E[j]
        return support
        # return self._elements

    def _insert(self, value, probability=None):
        """
        Updates this RV (its distribution)

        :param value: this is intended to be another Difference
        :param probability: probability of the other RV
        """
        if isinstance(value, self.__class__):
            # just updates the joint PMF for the moment...
            if probability is None: probability = 1.
            self._joint_pmf += probability*value._joint_pmf
        else:
            print("Not sure what you are doing, but still....")
            super()._insert(value, probability)
        # self._need_update = True

    def scalar_mult(self, other):
        raise NotImplementedError

    def sum(self, other):
        raise NotImplementedError


class CategoricalGame(StochasticGame):

    @staticmethod
    def from_logits_game(v, categories=None):

        if categories is None:
            l0 = v(set())  # this should be a 1 x d matrix
            categories = tuple(range(np.shape(l0)[1]))
        else:
            if not isinstance(categories, tuple):
                # convert to tuples for hashing
                categories = tuple(categories)

        def _payoff(S):
            _lst = tuple(CategoricalPayoff(logits, categories)
                         for logits in v(S))
            ary = np.array(_lst, dtype=object)
            return ary

        return CategoricalGame(_payoff)

# ---------------------------------------------------
# "transition" functions to create games
# from machine learning objects.
# ---------------------------------------------------


def game_from_ml_model_with_baseline(model: Callable[[np.array], np.array],
                                     x: np.array, baseline: np.array,
                                     game_output_fn=None):
    """
    Game from ml model with fixed or callable baseline.

    :param model: a model, as a callable. The model must support batch evaluation
    :param x: one data point
    :param baseline: a baseline. This can be one data point, a (scalar) constant or
                        a callable with that takes the in-coalition mask (binary matrix)
                        and returns a `np.array`
    :param game_output_fn: function to process the output of the game, e.g. for setting up a structured (probabilistic)
                            game. The function takes the model output computed on the coalitions as input.
    :return: returns a game
    """
    if x.ndim > 1:  # batching of examples not supported
        raise NotImplementedError()

    d = x.shape[0]

    def payoff(S):
        in_coalition_mask = np.zeros((len(S), d))
        for k, e in enumerate(S):
            in_coalition_mask[k, maybe_convert(e)] = 1

        b = maybe_call(baseline, in_coalition_mask)
        x_hat = in_coalition_mask * x + (1 - in_coalition_mask) * b  # array of dims [batch size, d]

        return maybe_call(game_output_fn, model(x_hat), True)

    return Game(payoff)


def game_from_ml_model_with_baseline_torch(model, x, baseline: np.array):
    """
    Same as above but for torch models
    """
    import torch
    if x.ndim > 1:  # batching not supported
        raise NotImplementedError()

    d = x.shape[0]

    def payoff(S):
        in_coalition_mask = torch.zeros((len(S), d), device=x.device)
        for k, e in enumerate(S):
            in_coalition_mask[k, maybe_convert(e)] = 1

        b = maybe_call(baseline, in_coalition_mask)
        x_hat = in_coalition_mask * x + (1 - in_coalition_mask) * b  # array of dims [batch size, d]

        return model(x_hat)

    return Game(payoff)
