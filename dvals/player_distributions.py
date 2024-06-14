"""
This file contains all that has to do with distributions over coalitions or permutations,
a component of probabilistic and random-order group values.

There are three type of main classes:
- Support
- CoalitionDistribution
- PermutationDistribution

Support classes allow iteration over coalitions or permutations that have positive probability to occur

Distribution classes further handle the probability of such coalitions or permutations.

-----

Form a value operator standpoint, it is meant to support 3 classes of value operators:
- Probabilistic group values via coalition distributions
- Random-order group values (aka asymmetric Shapley value) via permutation distribution
- Semi-values (not yet implemented)

Reference: Chapter 7 (Probabilistic values for games).
"""

import itertools
import math
from abc import ABC
from functools import reduce

import numpy as np

from . import utils


class Support:
    """Represents a discrete support for a discrete random variable. A support of an RV is the set of points (/events)
    for which the probability is positive (i.e. that have probability mass)
    """

    # We will use instances of these classes both for supports over coalitions and permutations.

    def __len__(self):
        """How many permutations or coalitions belong to this support"""
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()

    @property
    def players(self):
        """The base players that compose this support.
        Note: order is not guaranteed! """
        raise NotImplementedError()

    def __or__(self, other):
        """Chaining of two supports"""
        if other == 0:
            return self
        else:
            return self._union(other)

    def _union(self, other):
        """This works as the chaining of the two supports"""

        class _ChainedSupport(Support):

            def __init__(self, *supports):
                self._supports = supports

            def __len__(self):
                return sum([len(s) for s in self._supports])

            def __iter__(self):
                # does not check that there are no repeated players... I think this is fine, but it may not down
                # the line. To keep this in mind!
                return itertools.chain(*[iter(s) for s in self._supports])

            @property
            def players(self):
                return list(reduce(lambda p, n: p | set(n.players), self._supports, set()))

        return _ChainedSupport(self, other)


class ListSupport(list, Support, ABC):
    """This is an umbrella class for simple supports based on lists. It's here in case we can generalize
    methods from child classes. A child class that is missing at the moment is ListCoalitionSupport.
    Will add if needed."""

    def _union(self, other):
        if isinstance(other, ListSupport):
            return type(self)(self + other)
        else:
            raise NotImplementedError('+ op only implemented for ListSupport types')


class ListPermutationSupport(ListSupport):

    @property
    def players(self):
        """
        The players here are all the entities in any of the permutations
        """
        return self[0]

    def _union(self, other):
        if not isinstance(other, ListPermutationSupport):
            raise NotImplementedError()
        if set(self.players) != set(other.players):
            raise ValueError('The players of two permutation supports should coincide!')
        return super()._union(other)


class _PlayerBasedSupport(Support, ABC):
    """Supports for which players (players) are explicitly provided"""

    def __init__(self, players):
        if isinstance(players, int):
            players = list(range(players))
        elif isinstance(players, (list, tuple)):
            pass
        else:
            raise ValueError(f'{players} not recognized')
        # since we'll use players to compute values, it's safer to keep them ordered!
        self._players = players
        self._length = self._compute_length()

    def _compute_length(self):
        raise NotImplementedError()

    def __len__(self):
        return self._length

    @property
    def players(self):
        return self._players


class FactorialPermutationSupport(_PlayerBasedSupport):
    """Support over permutations that includes all possible permutations of the given players"""

    def __init__(self, players):
        super().__init__(players)

    def _compute_length(self):
        return math.factorial(len(self.players))

    def __iter__(self):
        return itertools.permutations(self._players)


class EmptySupport(Support):
    """Represents the empty set (useful for initialization)."""

    def __len__(self): return 0

    def __iter__(self): return iter([])

    @property
    def players(self): return []


class PowersetSupport(_PlayerBasedSupport):

    def __init__(self, players, starter=None):
        super().__init__(players)
        self.starter = starter

    def __iter__(self):
        it = map(set, utils.powerset(self._players))
        if self.starter:
            return map(lambda c: self.starter | c, it)
        else:
            return it

    def _compute_length(self):
        return 2 ** len(self.players)


# ----------- distributions ---------------


class Distribution:

    def __init__(self, support) -> None:
        super().__init__()
        self._support = support

    def sample(self, n_samples=1):
        raise NotImplementedError()

    def probability(self, S):
        """
        Returns the probability of a given object

        :param S: list or array of coalitions or permutations
        """
        raise NotImplementedError()

    @property
    def support(self):
        """
        Returns a `Support` iterator that represents the support of this distribution
        """
        return self._support

    def iter_over(self, batch_size=None):
        raise NotImplementedError()

    def __len__(self):
        return len(self.support)

    @property
    def players(self):
        """ALl the players (players) that are part of the distribution"""
        return self.support.players


class CoalitionDistribution(Distribution, ABC):

    def iter_over(self, batch_size=None):
        """
        For distribution over coalitions, the iterator returns triplets
        where the first element is a probability, the second element are the players
         and the third element is a batch
        of coalitions, one for each element of the support.

        Each coalition of the
        batch is relative to each player, and it is compatible with the
        "insertion-style" formula of group values (hence it will never contain
        that player).

        For instance, say there are 3 players {a b c}. Then each batch is
        made of 3 coalitions as:
        [{}, {}, {}]
        or
        [{b}, {a}, {a}]  etc.

        See also computational note in values._compute_global_for_one_coalition

        """
        raise NotImplementedError()



class _PowersetCD(CoalitionDistribution, ABC):

    def __init__(self, players) -> None:
        super().__init__(PowersetSupport(players))

        def _remove_and_return(player):
            res = list(self.support.players)
            res.remove(player)
            return res

        # this is the key
        self._multi_supports = [
            PowersetSupport(_remove_and_return(e)) for e in self.support.players
        ]

    def iter_over(self, batch_size=None):
        # attention! this returns 1 probability and an array of sets. the probability applies to all sets!
        return map(lambda sets: (self.probability(sets[0]), self.players, np.array(sets)),
                   zip(*self._multi_supports))

    def __len__(self):
        return len(self._multi_supports[0])


class ShapleyCD(_PowersetCD):

    def __init__(self, players) -> None:
        super().__init__(players)
        self.d = len(self.support.players)

    def sample(self, n_samples=1):
        """First draws uniformly a coalition size for each player and then draws
        uniformly a coalition of the drawn size. """
        if n_samples != 1: raise NotImplementedError()

        rng = utils.get_rng()

        def _one_batch():
            return [
                set(rng.choice(support.players,
                               rng.integers(0, self.d), replace=False)
                    ) for support in self._multi_supports
            ]

        return np.array(_one_batch())

    def probability(self, S):
        d, s = len(self.support.players), len(S)
        return 1 / (d * math.comb(d - 1, s))


class BanzhafCD(_PowersetCD):
    # Won't need this atm

    def sample(self, n_samples=1):
        """This is implemented as sampling each of the player
        with probability 1/2.

        Proof:

         1/2^n = P_{Banzhaf}(S) = \prod_{s\in S} p(s \in S) \prod p(s \not\in S)

         then p ~ Ber(1/2)
         """
        rng = utils.get_rng()

        # defer next line to the backend?
        result = np.array([set() for _ in range(n_samples)])

        for e in self.support.players:
            result |= rng.choice([{e}, set()], n_samples)
        return result

    def probability(self, S):
        return 1. / len(self.support)


class PermutationDistribution(Distribution):

    def __init__(self, permutation_support, probabilities=None) -> None:
        """
        Distribution over permutations.

        :param permutation_support: Support over permutations
        :param probabilities: A list of probability values (must sum to 1). If None (default) assumes uniform
                                probability.
        """
        super().__init__(permutation_support)
        if probabilities:
            self._probabilities = probabilities
            self._const_p = None
        else:  # assume uniform distribution
            try:
                self._const_p = 1. / len(self.support)
            except OverflowError:
                print("Overflow error, use only sample!")
                self._const_p = None
            self._probabilities = None

    def sample(self, n_samples=1):
        #  maybe use repeat = False (even though strictly speaking that's no longer an
        #  unbiased sample). In this case, check that n_saample <= len(support)!
        rng = utils.get_rng()
        if isinstance(self.support, ListSupport):
            perms = rng.choice(self.support, (n_samples,), p=self._probabilities)
        elif isinstance(self.support, FactorialPermutationSupport):
            ip = [rng.permutation(len(self.players)) for _ in range(n_samples)]
            perms = [tuple(self.players[i] for i in p) for p in ip]
            if n_samples == 1: perms = perms[0]
        else:
            raise NotImplementedError(f"No implementation for support of type {type(self.support)}")
        return perms

    def probability(self, pi):
        if self._const_p:
            return self._const_p
        else:
            return self._probabilities[self.support.index(pi)]

    def iter_over(self, batch_size=1):
        """Yields tuples of probability, object from support"""
        if batch_size > 1:
            raise NotImplementedError()
        return map(lambda e: (self.probability(e), e), self.support)

    @staticmethod
    def natural_order(players, inverse=False):
        """
        Returns a probability distribution over permutation following the order of the `players`.
        `inverse=True` reverses the order.
        """
        if isinstance(players, int):
            players = list(range(players))
        if inverse:
            players = list(reversed(players))
        return PermutationDistribution(ListPermutationSupport([players]))

    @staticmethod
    def deltas(*permutations, probabilities=None):
        """
        Constructs a sum-of-delta distribution over permutations with optionally
        specified `probabilities` (otherwise uniform)

        :param permutations: one or multiple permutations
        :param probabilities: [optional] a list of probabilities, one for each permutation
        :return: a `PermutationDistribution` object
        """
        return PermutationDistribution(ListPermutationSupport(permutations),
                                       probabilities=probabilities)


    @staticmethod
    def shapley(players):
        return PermutationDistribution(FactorialPermutationSupport(players))
