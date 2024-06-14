"""
This file contains code for computing or estimating probabilistic and random-order group values..
"""

from typing import Optional

import numpy as np
from tqdm import tqdm

from .player_distributions import PermutationDistribution, CoalitionDistribution
from .games import StochasticGame


class GroupValues(dict):
    """Stores group values computed with `values.compute` or `values.approximate` functions.

    This is a dictionary class with some added functionalities.
    The dictionary keys are players and dictionary values are the players' worth (or value)
    """

    @property
    def offset(self):
        return self._offset if hasattr(self, '_offset') else None

    @property
    def grand_payoff(self):
        return self._grand_payoff if hasattr(self, '_grand_payoff') else None

    # noinspection PyUnresolvedReferences
    @property
    def players(self):
        return self._players

    # noinspection PyAttributeOutsideInit
    def set_name(self, name):
        """Set a name and return self"""
        self._name = name
        return self

    @property
    def name(self):
        if hasattr(self, '_name'):
            return self._name
        return 'Unnamed'

    # noinspection PyAttributeOutsideInit
    def set_n_samples(self, n_samples):
        self._n_samples = n_samples
        return self

    @property
    def n_samples(self):
        if hasattr(self, '_n_samples'):
            return self._n_samples
        return None

    def np_values(self, include_offset=False, include_gp=False, include_grand_diff=False, normalized=False):
        """
        :return: the group values as a numpy array
        """
        lst = list([v for k, v in self.items()])
        if include_offset: lst.append(self.offset)
        if include_gp: lst.append(self.grand_payoff)
        if include_grand_diff: lst.append(self.grand_payoff - self.offset)
        ary = np.stack(lst)
        if ary.ndim == 2 and ary.shape[1] == 1:
            ary = ary.reshape(-1)
        return ary

    @classmethod
    def initialize(cls, game, players):
        empty_payoff, gc_payoff = game([set(), set(players)])
        phi = cls((e, 0.) for e in players)
        phi._offset = empty_payoff
        phi._grand_payoff = gc_payoff
        phi._players = players
        return phi

    def update_value(self, player, probability, val):
        self[player] += probability * val
        return self

    def _copy_additional_attributes(self, original):
        self._offset = original.offset
        self._grand_payoff = original.grand_payoff
        return self.set_name(original.name).set_n_samples(original.n_samples)

    def sort(self, key=None):
        if key is None:
            # assume keys are integers from 0 to n
            key = lambda k: k[0]  # if k[0] is not GroupValues.OFFSET_KEY else np.inf
        sorted_items = sorted(self.items(), key=key)
        return GroupValues(sorted_items)._copy_additional_attributes(self)

    def remap_keys(self, new_labels):
        """
        Gives new name to the keys.

        :param new_labels: list of new names, expected in order.
        :return: A new GroupValue object
        """
        gnv = self.__class__((nk, v) for nk, v in zip(new_labels, self.values()))
        return gnv._copy_additional_attributes(self)

    def map(self, map_f):
        gnv = GroupValues(map_f(k, v) for k, v in self.items())
        return gnv._copy_additional_attributes(self)

    def filter(self, filter_f):
        gnv = GroupValues((k, v) for k, v in self.items() if filter_f(k, v))
        return gnv._copy_additional_attributes(self)

    @staticmethod
    def aggregate(values, aggregator='mean', players=None):
        """
        Aggregates the group values from multiple objects, optionally specifying aggregator function and players

        :param values: List of GroupValues objects
        :param aggregator: str aggregator function
        :param players: optional list of players (e.g. if you want to aggregate for a subset of players)
        :return:
        """
        if players is None: players = values[0].keys()
        if aggregator == 'mean':
            op = np.mean
        else:
            raise NotImplementedError()
        return GroupValues((k, op([val[k] for val in values if k in val]))
                           for k in players)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __add__(self, other):
        if isinstance(other, GroupValues):
            return GroupValues((k, v + other[k]) for k, v in self.items())
        raise ValueError()

    def __rmul__(self, c):
        """
        Multiplication by a scalr, copies

        :param c: a scalr
        :return: group values multiplied by that scalar
        """
        return GroupValues((k, c * v) for k, v in self.items())



class DistributionalGroupValues(GroupValues):
    """Basic class for structured group values"""

    # this seems a sensible choice as it is very unlikely that there will
    # be any player named`Offset`

    @property
    def offset(self):
        return self._offset if hasattr(self, '_offset') else None

    @property
    def grand_payoff(self):
        return self._grand_payoff if hasattr(self, '_grand_payoff') else None

    # noinspection PyUnresolvedReferences
    @property
    def players(self):
        return self._players

    # ----- the core methods ------

    def mul(self, a, z):
        """Multiplication by scalar (optionally restricted) in the difference space"""
        raise NotImplementedError()

    def sum(self, z, w):
        """Sum in the difference space"""
        raise NotImplementedError()

    # def aggregated_values  # maybe in the future

    def np_values(self, include_offset=False, include_gp=False, include_grand_diff=False, normalized=False):
        """
        :return: the group values as a numpy array
        """
        lst = list([v for k, v in self.items()])
        if include_offset: lst.append(self.offset)
        if include_gp: lst.append(self.grand_payoff)
        if include_grand_diff: lst.append(self.grand_payoff - self.offset)
        op = lambda v: map(lambda c: c.probabilities, v)
        ary = np.stack(list(op(lst)))
        if ary.ndim == 2 and ary.shape[1] == 1:
            ary = ary.reshape(-1)
        return ary

    @classmethod
    def initialize(cls, game, players):
        empty_payoff, gc_payoff = game([set(), set(players)])
        phi = cls((e, gc_payoff.empty_diff()) for e in players)
        phi._offset, phi._grand_payoff = empty_payoff, gc_payoff
        phi._players = players
        return phi

    @classmethod
    def _init_dict(cls, players):
        gv = cls((e, None) for e in players)
        return gv

    def update_value(self, player, probability, val):
        self[player] |= (val, probability)  # this is fine for discrete games
        return self


def _initialize(game, players):
    if isinstance(game, StochasticGame):
        return DistributionalGroupValues.initialize(game, players)
    else:
        return GroupValues.initialize(game, players)

def compute(game, distribution, progress_bar=True):
    """
    Computes group values of `game` given a distribution over coalitions or over permutations.
    This is full, deterministic computation

    :param game: an instance of `Game`
    :param distribution: a distribution either over coalitions or over permutations.
                        In the first case, values are known as probabilistic group values
                        in the second, values are known as random group order values
                        [R. J. Weber. Probabilistic values for games. Cambridge University Press, 1988.]
    :param progress_bar: whether to show a progress bar (default: True)

    :return: a `GroupValue` object, that extend dict({player: value}).
    """
    phi = _initialize(game, distribution.players)

    iterator = tqdm(distribution.iter_over(), total=len(distribution)) if progress_bar else distribution.iter_over()

    if isinstance(distribution, PermutationDistribution):
        for probability, permutation in iterator:
            phi = _compute_for_one_permutation(game, permutation, probability, phi)

    elif isinstance(distribution, CoalitionDistribution):
        for probability, players, coalitions in iterator:  # probability(ies), players, coalitions
            phi = _compute_all_for_one_coalition(
                game, coalitions, players, probability, phi)

    else:
        raise NotImplementedError("The distribution must be a subclass of either"
                                  " PermutationDistribution or CoalitionDistribution")
    return phi


def estimate(game, distribution, n_samples, phi: Optional[GroupValues] = None, progress_bar=False):
    """
    Estimates group values of `game` given a distribution over coalitions or over permutations.

    :param game: an instance of `Game`
    :param distribution: a distribution either over coalitions or over permutations.
                        In the first case, values are known as probabilistic group values
                        in the second, values are known as random group order values
                        [R. J. Weber. Probabilistic values for games. Cambridge University Press, 1988.]
    :param phi: optional `GroupValue` object, if not None will restart estimate on the topo of `phi`
    :param n_samples: How many samples (of coalitions or permutations) to draw
    :param progress_bar: whether to show a progress bar (default: True)

    :return: a dictionary {player: value}. `None` stands for the empty set.
    """
    if phi is None:
        phi = _initialize(game, distribution.players)
        probability = None if isinstance(phi, DistributionalGroupValues) else 1./n_samples
        sum_ns = n_samples
    else:
        raise NotImplementedError

    iterator = tqdm(range(n_samples)) if progress_bar else range(n_samples)

    for _ in iterator:
        sample = distribution.sample()  # add batch_size!

        if isinstance(distribution, PermutationDistribution):
            phi = _compute_for_one_permutation(game, sample, probability, phi)

        elif isinstance(distribution, CoalitionDistribution):
            phi = _compute_all_for_one_coalition(
                game, sample, distribution.players, probability, phi)
        else:
            raise AttributeError("The distribution must be a subclass of either"
                                 " PermutationDistribution or CoalitionDistribution, "
                                 f"found {type(distribution)}")
    return phi.set_n_samples(sum_ns)


def _compute_for_one_permutation(game, pi, weight=None, phi=None):
    """
    Computes group values for one permutation pi

    :param game: a Game
    :param pi: a permutation (as a tuple, hence ordered!)
    :param weight: [optional] a group weight of the permutation
    :param phi: [optional] (partially computed) group values

    :return: a dictionary {player (of permutation): weight * value}
    """
    if phi is None: phi = _initialize(game, pi)  # pi is a list of players
    coalitions = np.array([set(pi[:i]) for i in range(1, len(pi) + 1)])

    phi_hat = game(coalitions)
    phi_hat[1:] = phi_hat[1:] - phi_hat[:-1]

    phi_hat[0] = phi_hat[0] - phi.offset

    for k, i in enumerate(pi):
        phi.update_value(i, weight, phi_hat[k])  # phi[i] += weight * phi_hat[k] (for standard GV)
    return phi


def _compute_all_for_one_coalition(game, coalitions, players, p=None, phi=None):
    if phi is None: phi = _initialize(game, players)
    if not isinstance(p, list):
        p = [p] * len(players)
    phi_hat = game(coalitions | np.array([{e} for e in players])) - game(coalitions)

    for k, ph, prb in zip(players, phi_hat, p):
        phi.update_value(k, prb, ph)  # phi[k] += prb * ph
    return phi
