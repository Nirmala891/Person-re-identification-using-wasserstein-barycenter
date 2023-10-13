#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import ot
import time
import numpy as np

from ot.bregman import sinkhorn
from msda.utils import barycentric_mapping
from msda.utils import semisupervised_penalty

from msda.ot_gpu import sinkhorn_gpu


def sinkhorn_barycenter(mu_s, Xs, Xbar, ys=None, ybar=None, reg=1e-3, b=None, weights=None,
                        method="sinkhorn", norm="max", metric="sqeuclidean", numItermax=100,
                        numInnerItermax=1000, stopThr=1e-4,
                        log=False, line_search=False, limit_max=np.infty, callbacks=None,
                        device='cuda:0', **kwargs):
   
    N = len(mu_s)
    k = Xbar.shape[0]
    d = Xbar.shape[1]
    if b is None:
        b = np.ones([k, ]) / k
    if weights is None:
        weights = [1 / N] * N

    displacement = stopThr + 1
    count = 0
    comp_start = time.time()
    log_dict = {'displacement_square_norms': [],
                'barycenter_coordinates': [Xbar]}
    old_Xbar = np.zeros([k, d])
    while (displacement > stopThr and count < numItermax):
        tstart = time.time()
        T_sum = np.zeros([k, d])
        transport_plans = []

        for i in range(N):
            Mi = ot.dist(Xs[i], Xbar, metric=metric)
            Mi = ot.utils.cost_normalization(Mi, norm=norm)
            if ys is not None and ybar is not None:
                Mi = semisupervised_penalty(ys=ys[i], yt=ybar, M=Mi, limit_max=limit_max)
            if implementation == 'torch':
                T_i = sinkhorn_gpu(mu_s[i], b, Mi, reg, numItermax=numInnerItermax, device=device)
            else:
                T_i = sinkhorn(mu_s[i], b, Mi, reg, numItermax=numInnerItermax, **kwargs)
            transport_plans.append(T_i.T)
        T_sum = sum([
            wi * barycentric_mapping(Xt=Xsi, coupling=Ti) for wi, Ti, Xsi in zip(weights, transport_plans, Xs)
        ])

        if line_search:
            alpha = naive_line_search(_barycenter_cost, Xbar, T_sum, args=(Xs, b, transport_plans), max_iter=21,
                                      )

            

        Xbar = (1 - alpha) * Xbar + alpha * T_sum
        displacement = np.sum(np.square(Xbar - old_Xbar))
        old_Xbar = Xbar.copy()
        tfinish = time.time()

        
    
   
        return Xbar, transport_plans

