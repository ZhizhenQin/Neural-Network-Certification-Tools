# -*- coding: utf-8 -*-
"""CBFverify_Lie.ipynb

* Codes adapted from https://github.com/huanzhang12/CertifiedReLURobustness and https://github.com/huanzhang12/RecurJac-and-CROWN

```
@inproceedings{weng2018CertifiedRobustness,
  author = "Tsui-Wei Weng AND Huan Zhang AND Hongge Chen AND Zhao Song AND Cho-Jui Hsieh AND Duane Boning AND Inderjit S. Dhillon AND　Luca Daniel",
  title = "Towards Fast Computation of Certified Robustness for ReLU Networks",
  booktitle = "International Conference on Machine Learning (ICML)",
  year = "2018",
  month = "july"
}
```

# Verify the conditions of CBF - part 2:
* Goal: $\dot B(x) < 0, \; \forall x \in \{x|B(x) = 0\}$

* Approach: Find $UB$ and $LB$ s.t. $LB \leq \dot B(x) \leq UB$ 
  * note: $\dot B(x) = \sum_i \frac{\partial B(x)}{\partial x_i} \frac{dx_i}{dt}$. In this notebook, we implement functions that can find the intervals of $\frac{\partial B(x)}{\partial x_i}$ only.  
  * $\frac{\partial B(x)}{\partial x_i} = \sum_k W_{1,k}^{(2)} \sigma^\prime(W_{k.:}^{(1)}x+b_k^{(1)}) W_{k.i}^{(1)} = W_{1,:}^{(2)} \left[ \sigma^\prime(W^{(1)}x+b^{(1)}) \odot W_{:,i}^{(1)} \right]$

* Implemented interval bounds:
  * get_lie_derivative_bounds - for multi-layer network with ReLU activation

* Usage: (support batch dimension for $x_0$)
```
UBs, LBs = get_lie_derivative_bounds(x0, eps, Ws, bs) # UBs & LBs shape: (num_batch, dim_in, 1)
```

"""

""" ## Import """
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from cbfverify_barrier_fastlin import get_weights
from cbfverify_barrier_fastlin_batch import get_barrier_bounds_fastlin_batch_mult_layers

import numpy as np
"""## Setup

### Lie derivative bound
"""

def get_derivative_bounds_mult_layers(x0, eps, Ws, bs, val_UBs=None, val_LBs=None, activation="relu"):
    assert x0.shape[1] == 1, "currently only support num_batch = 1 computation"
    
    num_layers = len(Ws)
    
    if val_UBs is None or val_LBs in None:
        val_UBs, val_LBs = get_barrier_bounds_fastlin_batch_mult_layers(x0, eps, Ws, bs, method="fastlin", activation=activation)
        
    neuron_states = get_neuron_states(val_UBs, val_LBs)
    
    Ws_np = [w.detach().cpu().numpy() for w in Ws]
    neuron_states_np = [ns.detach().cpu().numpy() for ns in neuron_states]
    
    UBs, LBs = derivative_bounds(Ws_np, neuron_states_np, num_layers, activation=activation)
    
    UBs = torch.from_numpy(UBs).T.unsqueeze(0)
    LBs = torch.from_numpy(LBs).T.unsqueeze(0)
    
    return UBs, LBs


def get_neuron_states(UBs, LBs):
    neuron_states = []
    for LB, UB in zip(LBs[1:-1], UBs[1:-1]):
        neuron_states.append(torch.zeros_like(UB))
        # neurons never activated set to -1
        neuron_states[-1] -= (UB < 0).long()
        # neurons always activated set to +1
        neuron_states[-1] += LB > 0
    return neuron_states


def derivative_bounds(weights, neuron_states, numlayer, activation="relu"):
    assert numlayer >= 2
    assert activation.lower() in ["relu"], "Activation {} is not supported".format(activation)
    # merge the last layer weights according to c and j
    # W_vec = np.expand_dims(weights[-1][c] - weights[-1][j], axis=0)
    # const, l, u = fast_compute_max_grad_norm_2layer(W_vec, weights[-2], neuron_states[-1])
    const, l, u = fastlip_2layer(weights[-1], weights[-2], neuron_states[-1])
    # for layers other than the last two layers
    for i in list(range(numlayer - 2))[::-1]:
        const, l, u = fastlip_nplus1_layer(const, l, u, weights[i], neuron_states[i])
    # get the final upper and lower bound
    l += const
    u += const
    
    
    return u, l


# W2 \in [c, M2], W1 \in [M2, M1]
# c, l, u \in [c, M1]
# r \in [c], k \in [M1], i \in [M2]
# @jit(nopython=True)
def fastlip_2layer(W2, W1, neuron_state, norm = 1):
    # even if q_n != 1, then algorithm is the same. The difference is only at the output of fast_compute_max_grad_norm
    assert norm == 1
    # diag = 1 when neuron is active
    diag = np.maximum(neuron_state.astype(np.float32), 0).T[0]
    unsure_index = np.nonzero(neuron_state == 0)[0]
    # this is the constant part
    c = np.dot(diag * W2, W1)
#     c = np.dot(np.dot(W2, diag), W1)
    # this is the delta, and l <=0, u >= 0
    l = np.zeros((W2.shape[0], W1.shape[1]), dtype=W2.dtype)
    u = np.zeros_like(l)
    for r in range(W2.shape[0]):
        for k in range(W1.shape[1]):
            for i in unsure_index:
                prod = W2[r,i] * W1[i,k]
                if prod > 0:
                    u[r,k] += prod
                else:
                    l[r,k] += prod
    return c, l, u


# prev_c is the constant part; prev_l <=0, prev_u >= 0
# prev_c, prev_l, prev_u \in [c, M2], W1 \in [M2, M1]
# r \in [c], k \in [M1], i \in [M2]
# @jit(nopython=True)
def fastlip_nplus1_layer(prev_c, prev_l, prev_u, W1, neuron_state, norm = 1):
    # c, l, u in shape(num_output_class, num_neurons_in_this_layer)
    c = np.zeros((prev_l.shape[0], W1.shape[1]), dtype = W1.dtype)
    l = np.zeros_like(c)
    u = np.zeros_like(c)
    # now deal with prev_l <= delta <= prev_u term
    # r is dimention for delta.shape[0]
    for r in range(prev_l.shape[0]):
        for k in range(W1.shape[1]):
            for i in range(W1.shape[0]):
                # unsure neurons
                if neuron_state[i] == 0:
                    if W1[i,k] > 0:
                        if W1[i,k] * (prev_c[r,i] + prev_u[r,i]) > 0:
                            u[r,k] += W1[i,k] * (prev_c[r,i] + prev_u[r,i])
                        if W1[i,k] * (prev_c[r,i] + prev_l[r,i]) < 0:
                            l[r,k] += W1[i,k] * (prev_c[r,i] + prev_l[r,i])
                    if W1[i,k] < 0:
                        if W1[i,k] * (prev_c[r,i] + prev_l[r,i]) > 0:
                            u[r,k] += W1[i,k] * (prev_c[r,i] + prev_l[r,i])
                        if W1[i,k] * (prev_c[r,i] + prev_u[r,i]) < 0:
                            l[r,k] += W1[i,k] * (prev_c[r,i] + prev_u[r,i])
                # active neurons
                if neuron_state[i] > 0:
                    # constant terms
                    c[r,k] += W1[i,k] * prev_c[r,i]
                    # upper/lower bounds terms
                    if W1[i,k] > 0:
                        u[r,k] += prev_u[r,i] * W1[i,k] 
                        l[r,k] += prev_l[r,i] * W1[i,k]
                    else:
                        u[r,k] += prev_l[r,i] * W1[i,k] 
                        l[r,k] += prev_u[r,i] * W1[i,k]
    return c, l, u



"""## Testing
### 1. random samples $x_{samp}$, check $LB_i \leq \dot B_i(x_{samp}) \leq UB_i$ for all $i$ and $x_{samp}$
"""

def generate_random_sample(x0, eps):
  # input: [dim, batch]
  # rand: [0,1) --> make input range: (-1, 1)
  x_samples = eps*(2*torch.rand(x0.shape)-1)+x0  # x_sample: shape (dim_in, num_batch)
  return x_samples


"""### 2. Ground truth derivatives"""
def forward_prop_derivative(x0, model):
    x0_in = x0.T
    x0_in = x0_in.requires_grad_(True)
    grads = torch.autograd.grad(model(x0_in),x0_in)
    
    return grads[0].T.unsqueeze(0)


"""### 3. Wrap as a function"""

def validate_derivative_bounds(x0, eps, Ws, bs, model, bound_func):
  # get bounds
  UBs, LBs = bound_func(x0, eps, Ws, bs)

  # generate MC samples
  x_samples = generate_random_sample(x0, eps)
  out_samples = forward_prop_derivative(x0, model)
   
  bounds = torch.cat((LBs[0],UBs[0]),1) # bounds[i]: (LB_i, UB_i)
  # show(out_samples[:,0,0].numpy(),out_samples[:,1,0].numpy(),bounds[0].numpy(),bounds[1].numpy())

  # check violations
  violation_UBs = torch.where(UBs < out_samples)
  violation_LBs = torch.where(LBs > out_samples)

  # print("UBs[8]:{}, LBs[8]:{}".format(UBs[8],LBs[8]))
  # print("out_samples[:,0,0]:{}".format(out_samples[:,0,0]))

  for i in range(len(violation_UBs)):
    assert torch.sum(violation_UBs[i]) == 0, "violating UBs[{}]".format(i)  
#   print("pass validation of UB!")

  for i in range(len(violation_LBs)):
    assert torch.sum(violation_LBs[i]) == 0, "violating LBs[{}]".format(i)
#   print("pass validation of LB!") 

  return violation_UBs, violation_LBs
  
  