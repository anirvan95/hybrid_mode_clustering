# Copyright (c) 2008 Carnegie Mellon University
#
# You may copy and modify this freely under the same terms as
# Sphinx-III

"""
Divergence and distance measures for multivariate Gaussians and
multinomial distributions.

This module provides some functions for calculating divergence or
distance measures between distributions, or between one distribution
and a codebook of distributions.
"""

__author__ = "David Huggins-Daines <dhuggins@cs.cmu.edu>"
__version__ = "$Revision$"

import numpy


def gau_bh(pm, pv, qm, qv):
    """
    Classification-based Bhattacharyya distance between two Gaussians
    with diagonal covariance.  Also computes Bhattacharyya distance
    between a single Gaussian pm,pv and a set of Gaussians qm,qv.
    """
    # Difference between means pm, qm
    diff = qm - pm
    # Interpolated variances
    pqv = (pv + qv) / 2.
    # Log-determinants of pv, qv
    ldpv = numpy.linalg.det(pv)
    ldqv = numpy.linalg.det(qv)
    # Log-determinant of pqv
    ldpqv = numpy.linalg.det(pqv)
    # "Shape" component (based on covariances only)
    # 0.5 log(|\Sigma_{pq}| / sqrt(\Sigma_p * \Sigma_q)
    norm = 0.5 * numpy.log(ldpqv/(numpy.sqrt(ldpv*ldqv)))
    # "Divergence" component (actually just scaled Mahalanobis distance)
    # 0.125 (\mu_q - \mu_p)^T \Sigma_{pq}^{-1} (\mu_q - \mu_p)
    dist = 0.125 * (diff * (1. / pqv) * diff)
    return dist + norm


def gau_kl(pm, pv, qm, qv):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
    """
    if (len(qm.shape) == 2):
        axis = 1
    else:
        axis = 0
    # Determinants of diagonal covariances pv, qv
    dpv = pv.prod()
    dqv = qv.prod(axis)
    # Inverse of diagonal covariance qv
    iqv = 1. / qv
    # Difference between means pm, qm
    diff = qm - pm
    return (0.5 *
            (numpy.log(dqv / dpv)  # log |\Sigma_q| / |\Sigma_p|
             + (iqv * pv).sum(axis)  # + tr(\Sigma_q^{-1} * \Sigma_p)
             + (diff * iqv * diff).sum(axis)  # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
             - len(pm)))  # - N


def kl_mvn(m0, S0, m1, S1):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.

    - accepts stacks of means, but only one S0 and S1

    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| +
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = numpy.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term = numpy.trace(iS1 @ S0)
    det_term = numpy.log(numpy.linalg.det(S1) / numpy.linalg.det(S0))  # np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ numpy.linalg.inv(
        S1) @ diff  # np.sum( (diff*diff) * iS1, axis=1)   #print(tr_term,det_term,quad_term)
    return .5 * (tr_term + det_term + quad_term - N)


def gau_js(pm, pv, qm, qv):
    """
    Jensen-Shannon divergence between two Gaussians.  Also computes JS
    divergence between a single Gaussian pm,pv and a set of Gaussians
    qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
    """
    if (len(qm.shape) == 2):
        axis = 1
    else:
        axis = 0
    # Determinants of diagonal covariances pv, qv
    dpv = pv.prod()
    dqv = qv.prod(axis)
    # Inverses of diagonal covariances pv, qv
    iqv = 1. / qv
    ipv = 1. / pv
    # Difference between means pm, qm
    diff = qm - pm
    # KL(p||q)
    kl1 = (0.5 *
           (numpy.log(dqv / dpv)  # log |\Sigma_q| / |\Sigma_p|
            + (iqv * pv).sum(axis)  # + tr(\Sigma_q^{-1} * \Sigma_p)
            + (diff * iqv * diff).sum(axis)  # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
            - len(pm)))  # - N
    # KL(q||p)
    kl2 = (0.5 *
           (numpy.log(dpv / dqv)  # log |\Sigma_p| / |\Sigma_q|
            + (ipv * qv).sum(axis)  # + tr(\Sigma_p^{-1} * \Sigma_q)
            + (diff * ipv * diff).sum(axis)  # + (\mu_q-\mu_p)^T\Sigma_p^{-1}(\mu_q-\mu_p)
            - len(pm)))  # - N
    # JS(p,q)
    return 0.5 * (kl1 + kl2)


def multi_kl(p, q):
    """Kullback-Liebler divergence from multinomial p to multinomial q,
    expressed in nats."""
    if (len(q.shape) == 2):
        axis = 1
    else:
        axis = 0
    # Clip before taking logarithm to avoid NaNs (but still exclude
    # zero-probability mixtures from the calculation)
    return (p * (numpy.log(p.clip(1e-10, 1))
                 - numpy.log(q.clip(1e-10, 1)))).sum(axis)


def multi_js(p, q):
    """Jensen-Shannon divergence (symmetric) between two multinomials,
    expressed in nats."""
    if (len(q.shape) == 2):
        axis = 1
    else:
        axis = 0
    # D_{JS}(P\|Q) = (D_{KL}(P\|Q) + D_{KL}(Q\|P)) / 2
    return 0.5 * ((q * (numpy.log(q.clip(1e-10, 1))
                        - numpy.log(p.clip(1e-10, 1)))).sum(axis)
                  + (p * (numpy.log(p.clip(1e-10, 1))
                          - numpy.log(q.clip(1e-10, 1)))).sum(axis))
