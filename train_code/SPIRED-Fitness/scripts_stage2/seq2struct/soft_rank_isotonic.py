import numpy as np
from numba import njit


@njit
def isotonic_l2(y, sol):
    """Solves an isotonic regression problem using PAV.

    Formally, it solves argmin_{v_1 >= ... >= v_n} 0.5 ||v - y||^2.

    Args:
      y: input to isotonic regression, a 1d-array.
      sol: where to write the solution, an array of the same size as y.
    """
    n = y.shape[0]
    target = np.arange(n)
    c = np.ones(n)
    sums = np.zeros(n)

    # target describes a list of blocks.  At any time, if [i..j] (inclusive) is
    # an active block, then target[i] := j and target[j] := i.

    for i in range(n):
        sol[i] = y[i]
        sums[i] = y[i]

    i = 0
    while i < n:
        k = target[i] + 1
        if k == n:
            break
        if sol[i] > sol[k]:
            i = k
            continue
        sum_y = sums[i]
        sum_c = c[i]
        while True:
            # We are within an increasing subsequence.
            prev_y = sol[k]
            sum_y += sums[k]
            sum_c += c[k]
            k = target[k] + 1
            if k == n or prev_y > sol[k]:
                # Non-singleton increasing subsequence is finished,
                # update first entry.
                sol[i] = sum_y / sum_c
                sums[i] = sum_y
                c[i] = sum_c
                target[i] = k - 1
                target[k - 1] = i
                if i > 0:
                    # Backtrack if we can.  This makes the algorithm
                    # single-pass and ensures O(n) complexity.
                    i = target[i - 1]
                # Otherwise, restart from the same point.
                break

    # Reconstruct the solution.
    i = 0
    while i < n:
        k = target[i] + 1
        sol[i + 1 : k] = sol[i]
        i = k


@njit
def _log_add_exp(x, y):
    """Numerically stable log-add-exp."""
    larger = max(x, y)
    smaller = min(x, y)
    return larger + np.log1p(np.exp(smaller - larger))


# Modified implementation for the KL geometry case.
@njit
def isotonic_kl(y, w, sol):
    """Solves isotonic optimization with KL divergence using PAV.

    Formally, it solves argmin_{v_1 >= ... >= v_n} <e^{y-v}, 1> + <e^w, v>.

    Args:
      y: input to isotonic optimization, a 1d-array.
      w: input to isotonic optimization, a 1d-array.
      sol: where to write the solution, an array of the same size as y.
    """
    n = y.shape[0]
    target = np.arange(n)
    lse_y_ = np.zeros(n)
    lse_w_ = np.zeros(n)

    # target describes a list of blocks.  At any time, if [i..j] (inclusive) is
    # an active block, then target[i] := j and target[j] := i.

    for i in range(n):
        sol[i] = y[i] - w[i]
        lse_y_[i] = y[i]
        lse_w_[i] = w[i]

    i = 0
    while i < n:
        k = target[i] + 1
        if k == n:
            break
        if sol[i] > sol[k]:
            i = k
            continue
        lse_y = lse_y_[i]
        lse_w = lse_w_[i]
        while True:
            # We are within an increasing subsequence.
            prev_y = sol[k]
            lse_y = _log_add_exp(lse_y, lse_y_[k])
            lse_w = _log_add_exp(lse_w, lse_w_[k])
            k = target[k] + 1
            if k == n or prev_y > sol[k]:
                # Non-singleton increasing subsequence is finished,
                # update first entry.
                sol[i] = lse_y - lse_w
                lse_y_[i] = lse_y
                lse_w_[i] = lse_w
                target[i] = k - 1
                target[k - 1] = i
                if i > 0:
                    # Backtrack if we can.  This makes the algorithm
                    # single-pass and ensures O(n) complexity.
                    i = target[i - 1]
                # Otherwise, restart from the same point.
                break

    # Reconstruct the solution.
    i = 0
    while i < n:
        k = target[i] + 1
        sol[i + 1 : k] = sol[i]
        i = k
