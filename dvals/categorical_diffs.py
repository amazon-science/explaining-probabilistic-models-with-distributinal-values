import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.reciprocal(np.exp(-x) + 1.0)


def joint_probability(
    alpha: np.ndarray, beta: np.ndarray, diag_only: bool = False
) -> dict:
    """
    Computes 2D joint probability table related to marginal contributions in
    games based on d-way categorical distributions. We return a dictionary
    with:
    * "joint": Joint probability matrix (shape ``(n, d, d)``;
      if ``diag_only == False``)
    * "diag": Diagonal of joint probability matrix (shape ``(n, d)``;
      if ``diag_only == True``)
    * "marg_left": Marginal probability vector over left variable
      (shape ``(n, d)``)
    * "marg_right": Marginal probability vector over left variable
      (shape ``(n, d)``)
    Note that the matrix has a simple representation in terms of ``O(d)``
    memory and compute, and many useful functions of the full table can
    likely be computed in less than ``O(d^2)`` space and time.
    :param alpha: Input vectors, shape ``(n, d)``
    :param beta: Input vectors, shape ``(n, d)``
    :param diag_only: If ``True``, only the diagonal of the joint probability
        matrix is computed, which costs ``O(d)`` only. Defaults to ``False``
    :return: Result dictionary (see above)
    """

    if alpha.ndim == 1:
        alpha = alpha.reshape((1, -1))
    assert alpha.size == beta.size
    beta = beta.reshape(alpha.shape)
    num_cases, num_categs = alpha.shape
    assert num_categs >= 3, "Must have at least 3 categories"
    # Reorder categories so that ``delta`` is nonincreasing
    delta = alpha - beta
    order_ind = np.argsort(-delta, axis=1)
    alpha_ord = np.take_along_axis(alpha, order_ind, axis=1)
    beta_ord = np.take_along_axis(beta, order_ind, axis=1)
    delta = np.take_along_axis(delta, order_ind, axis=1)
    # Compute :code:`log(sum(exp(alpha[:k])))` for all k<d. This is done by
    # applying :code:`accumulate` (which generalizes cumsum and cumprod) to the
    # ufunc
    #    ``(a, b) -> log(exp(a) + exp(b))``
    bar_alpha = np.logaddexp.accumulate(alpha_ord, axis=1)
    bar_alpha_d = bar_alpha[:, -1:]  # Needed for diagonal below
    bar_alpha = bar_alpha[:, :-1]
    bar_beta = np.logaddexp.accumulate(np.flip(beta_ord, axis=1), axis=1)
    bar_beta_0 = bar_beta[:, -1:]  # Needed below
    bar_beta = np.flip(bar_beta[:, :-1], axis=1)
    bar_diff = bar_beta - bar_alpha
    sigma_delta_k = sigmoid(bar_diff + delta[:, :-1])
    prob_diag_ord = np.concatenate(
        (
            sigma_delta_k * np.exp(beta_ord[:, :-1] - bar_beta),
            np.exp(alpha_ord[:, -1:] - bar_alpha_d),
        ),
        axis=1,
    )
    if not diag_only:
        sigma_delta_kp1 = sigmoid(bar_diff + delta[:, 1:])
        cvecs = np.exp(-bar_alpha - bar_beta) * (sigma_delta_k - sigma_delta_kp1)
        cumsum_c_ord = np.concatenate(
            (np.zeros((num_cases, 1)), np.cumsum(cvecs, axis=1)), axis=1
        )
    # Undo the reordering
    if not diag_only:
        cumsum_c = np.empty_like(cumsum_c_ord)
        np.put_along_axis(cumsum_c, indices=order_ind, values=cumsum_c_ord, axis=1)
    prob_diag = np.empty_like(prob_diag_ord)
    np.put_along_axis(prob_diag, indices=order_ind, values=prob_diag_ord, axis=1)
    result = {
        "marg_left": np.exp(alpha - bar_alpha_d),
        "marg_right": np.exp(beta - bar_beta_0),
    }
    if diag_only:
        result["diag"] = prob_diag
    else:
        # Compose probability matrix. Note that up to here, all computations are
        # O(d)
        inv_order_ind = np.zeros_like(order_ind)
        np.put_along_axis(
            inv_order_ind,
            indices=order_ind,
            values=np.arange(num_categs).reshape((1, -1)),
            axis=1,
        )
        r_shp = (num_cases, -1, 1)
        s_shp = (num_cases, 1, -1)
        prob_mat = (
            np.reshape(np.exp(alpha), r_shp)
            * (cumsum_c.reshape(s_shp) - cumsum_c.reshape(r_shp))
            * np.reshape(np.exp(beta), s_shp)
            * (inv_order_ind.reshape(r_shp) < inv_order_ind.reshape(s_shp))
        )
        ind1 = np.arange(num_cases).reshape((-1, 1))[:, [0] * num_categs].flatten()
        ind2 = np.arange(num_categs).reshape((1, -1))[[0] * num_cases].flatten()
        prob_mat[(ind1, ind2, ind2)] = prob_diag.flatten()
        result["joint"] = prob_mat
    return result


def query_max_probability_of_change(
    alpha: np.ndarray, beta: np.ndarray
) -> (np.ndarray, np.ndarray):
    """
    Computes the score for the query functional
    .. math::
       \mathrm{max}_s \mathbb{P}( v(S) = s, v(S\cup i) \ne s )
    Note: This is computing the query functional values for several different
    values of S (the first dimension, size ``n`, is over different S).
    What might be better instead is to compute the probability distributions
    for many different S, then average these to obtain a single distribution,
    and then compute the query functional based on this mixture distribution.
    :param alpha: Input vector, shape ``(n, d)``
    :param beta: Input vector, shape ``(n, d)``
    :return: Tuple of score value and argmax, both shape ``(n,)``
    """
    result = joint_probability(alpha, beta, diag_only=True)
    argument = result["marg_right"] - result["diag"]
    pos = np.argmax(argument, axis=1)
    maxvals = np.take_along_axis(
        argument, np.expand_dims(pos, axis=-1), axis=-1
    ).reshape((-1,))
    return maxvals, pos


def test_multi_inputs():
    num_categs = 5
    num_cases = 4
    alpha = np.random.normal(size=(num_cases, num_categs))
    beta = np.random.normal(size=(num_cases, num_categs))
    for diag_only in [True, False]:
        result1 = joint_probability(alpha, beta, diag_only=diag_only)
        result2 = [
            joint_probability(alpha=alpha[i], beta=beta[i], diag_only=diag_only)
            for i in range(num_cases)
        ]
        for i in range(num_cases):
            keys = ["marg_left", "marg_right"]
            if diag_only:
                keys.append("diag")
            else:
                keys.append("joint")
            for k in keys:
                np.testing.assert_almost_equal(
                    result1[k][i].flatten(), result2[i][k].flatten(), decimal=4
                )
