# -*- coding: utf-8 -*-
"""CBFverify_Barrier_fastlin.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1J9hroHOc-zLoBSNLUrgmkYKmrpuFGdz5

# Verify the conditions of CBF - part 1:
* Goal: Find $B(x) = 0$
* Approach: Find $UB$ and $LB$ s.t. $LB \leq B(x) \leq UB$ 
* Implemented both IBP and CROWN-general:
  * get_barrier_bounds_ibp
  * get_barrier_bounds_fastlin
* Usage for ibp bounds (support batch dimension for $x_0$)
```
UBs, LBs = get_barrier_bounds_ibp(x0, eps, Ws, bs, activation="tanh")
UBs[-1], LBs[-1] # the final output bounds
```

* Usage for crown bounds with tanh activation:
```
UBs, LBs = get_barrier_bounds_fastlin(x0, eps, Ws, bs, method="crown", activation="tanh")
UBs[-1], LBs[-1] # the final output bounds
```

"""

"""## Import"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


"""## Setup"""

def get_weights(net):
  Ws, bs = [], []
  state_dict = net.state_dict()
  cnt_w, cnt_b = 0, 0
  for key in state_dict.keys():
    val = state_dict[key]
    if "weight" in key:
      Ws.append(val)
      print("W"+str(cnt_w)+":"+str(val.shape))
      cnt_w += 1
    elif "bias" in key:
      bs.append(val.unsqueeze(1)) # make the bias dimension [d, 1]
      print("b"+str(cnt_b)+":"+str(val.shape))
      cnt_b += 1
    else:
      raise ValueError("layer not supported yet")
  assert len(Ws) == len(bs)
  return Ws, bs


"""## Get ibp bounds"""

def get_barrier_bounds_ibp(x0, eps, Ws, bs, activation="relu"):
  # x0: (dim_in, num_batch)
  # initial setting
  if activation == "relu":
    act = lambda x: torch.relu(x)
  elif activation == "tanh":
    act = lambda x: torch.tanh(x) 
  else:
    raise ValueError("the activation function is not supported yet")
    
  num_layer = len(Ws)
  assert len(Ws) == len(bs), "len(Ws) != len(bs)"
  UBs, LBs = [], []
  if type(eps) is torch.Tensor:
      eps = eps.unsqueeze(0).T
  UBs.append(x0+eps) # (dim_in, num_batch) 
  LBs.append(x0-eps) 

  for i in range(num_layer):
    W_pos=Ws[i].clone() # Ws[0]: (num_hidden, dim_in)
    W_neg=Ws[i].clone() # Ws[1]: (dim_out, num_hidden)
    W_pos[Ws[i]<0]=0
    W_neg[Ws[i]>0]=0

    UB=torch.matmul(W_pos,UBs[i])+torch.matmul(W_neg,LBs[i])+bs[i] # i=0, (num_hidden, num_batch)
    LB=torch.matmul(W_pos,LBs[i])+torch.matmul(W_neg,UBs[i])+bs[i] # i=1, (dim_out, num_batch)
    # print("i = {}, LB.shape={}".format(i, LB.shape))

    if i < num_layer-1: # last layer no activation
      UB = act(UB)
      LB = act(LB)
    UBs.append(UB)
    LBs.append(LB)
  
  return UBs, LBs 


"""## Get Fastlin bound"""

def solve_d_UB(u,l,act,derivative):
  max_iter = 10
  ub = u
  lb = 0
  d = u/2
  residual = d-act(l)-(d-l)*derivative(d)
  for i in range(max_iter):
    if d > 0 and torch.abs(residual) < 0.01:
      break;
    elif residual > 0:
      ub = d
      d = (d+lb)/2 # decrease d
    else: #residual < 0:
      lb = d
      d = (ub+d)/2 # increase d
  return d

def A_U_uns(UB,LB,act,derivative):
  # LB shape: (num_hidden, batch=1)
  d = torch.Tensor(UB.shape)
  # print("UB shape: {}".format(UB.shape))
  for i in range(LB.shape[0]):
    # solve d-act(l) = (d-l)*derivative(d)
    d[i] = solve_d_UB(UB[i],LB[i],act,derivative)
  
  A = derivative(d)
  
  return A

def solve_d_LB(u,l,act,derivative):
  max_iter = 10
  ub = 0
  lb = l
  d = l/2
  residual = act(d)-act(u)-(d-u)*derivative(d)
  for i in range(max_iter):
    if d < 0 and torch.abs(residual) < 0.01:
      break;
    elif residual > 0:
      ub = d
      d = (d+lb)/2 # decrease d
    else: #residual < 0:
      lb = d
      d = (ub+d)/2 # increase d
  return d

def A_L_uns(UB,LB,act,derivative):
  # LB shape: (num_hidden, batch=1)
  d = torch.Tensor(UB.shape)
  # print("UB shape: {}".format(UB.shape))
  for i in range(LB.shape[0]):
    # solve d-act(l) = (d-l)*derivative(d)
    d[i] = solve_d_LB(UB[i],LB[i],act,derivative)
  
  A = derivative(d)
  
  return A

def get_upper_coefficients(UB,LB,activation="relu"):
  # UB shape: (num_hidden, batch=1)
  idx_pos = LB>0
  idx_neg = UB<0
  idx_uns = torch.logical_and(LB<=0,UB>=0)

  # A_U, B_U
  # A_U: (num_hidden, num_hidden, num_batch)
  # B_U: (num_hidden, num_batch)
  # A_U_diag: (num_hidden, num_batch)
  A_U_diag = torch.zeros(UB.shape[0], UB.shape[1])
  B_U = torch.zeros(UB.shape[0],UB.shape[1])

  if activation=="relu":
    A_U_diag[idx_pos] = 1
    A_U_diag[idx_uns] = UB[idx_uns]/(UB[idx_uns]-LB[idx_uns])
    B_U[idx_uns] = -A_U_diag[idx_uns]*LB[idx_uns]
  # elif activation=="tanh":
  #   act = lambda x: torch.tanh(x)
  #   derivative = lambda x: 1-torch.tanh(x)**2
  #   # 1. idx_pos
  #   d = 0.5*(LB[idx_pos]+UB[idx_pos])
  #   A_U_diag[idx_pos] = derivative(d)
  #   B_U[idx_pos] = act(d) - d*derivative(d)

  #   # 2. idx_neg
  #   A_U_diag[idx_neg] = (act(UB[idx_neg])-act(LB[idx_neg]))/(UB[idx_neg]-LB[idx_neg])
  #   B_U[idx_neg] = act(LB[idx_neg])-A_U_diag[idx_neg]*LB[idx_neg]

  #   # 3. idx_uns
  #   A_U_diag[idx_uns] = A_U_uns(UB[idx_uns],LB[idx_uns],act,derivative)
  #   B_U[idx_uns] = -A_U_diag[idx_uns]*LB[idx_uns] + act(LB[idx_uns])

  else:
    raise ValueError("the activation function is not supported yet!")

  # A_U: (num_batch, num_hidden, num_hidden)
  A_U = torch.diag_embed(A_U_diag.T)
  # A_U = A_U.permute((2, 1, 0))

  return A_U, B_U

def get_lower_coefficients(UB,LB,activation="relu",method="fastlin"):
  # UB shape: (num_hidden, batch=1)
  idx_pos = LB>0
  idx_neg = UB<0
  idx_uns = torch.logical_and(LB<=0,UB>=0)

  # A_U, B_U
  # A_U: (num_hidden, num_hidden, num_batch)
  # B_U: (num_hidden, num_batch)
  # A_U_diag: (num_hidden, num_batch)
  A_L_diag = torch.zeros(UB.shape[0],UB.shape[1])
  B_L = torch.zeros(UB.shape[0],UB.shape[1])

  if activation=="relu":
    A_L_diag[idx_pos] = 1


  if method=="fastlin":
    assert activation=="relu", "fast-lin only implements relu"
    A_L_diag[idx_uns] = UB[idx_uns]/(UB[idx_uns]-LB[idx_uns])
  elif method=="crown":
    assert activation=="relu", "crown only (currently) implements relu"
    if activation=="relu":
      idx_UgeqL = torch.logical_and(idx_uns,UB>=torch.abs(LB))
      A_L_diag[idx_UgeqL] = 1
  #   elif activation=="tanh":
  #     act = lambda x: torch.tanh(x)
  #     derivative = lambda x: 1-torch.tanh(x)**2

  #     # 1. idx_pos
  #     A_L_diag[idx_pos] = (act(UB[idx_pos])-act(LB[idx_pos]))/(UB[idx_pos]-LB[idx_pos])
  #     B_L[idx_pos] = act(LB[idx_pos])-A_L_diag[idx_pos]*LB[idx_pos]

  #     # 2. idx_neg
  #     d = 0.5*(LB[idx_neg]+UB[idx_neg])
  #     A_L_diag[idx_neg] = derivative(d)
  #     B_L[idx_neg] = act(d) - d*derivative(d)

  #     # 3. idx_uns
  #     A_L_diag[idx_uns] = A_L_uns(UB[idx_uns],LB[idx_uns],act,derivative)
  #     B_L[idx_uns] = -A_L_diag[idx_uns]*UB[idx_uns] + act(UB[idx_uns])
  #   else:
  #     raise ValueError("the activation function is not supported yet!")
  # elif method=="crownzero":
  #   assert activation=="relu", "crown-zero only implements relu"
  #   pass # because coefficients = 0
  else:
    raise ValueError("unknown method in get_lower_coefficients!")

  # A_L: (num_batch, num_hidden, num_hidden)
  A_L = torch.diag_embed(A_L_diag.T)
  # A_L = A_L.permute((2, 1, 0))

  return A_L, B_L

def get_barrier_bounds_fastlin_batch(x0, eps, Ws, bs, method="fastlin", activation="relu"):
  # x0: (dim_in, num_batch)
  # initial setting
  # act = lambda x: torch.relu(x) 
  num_layer = len(Ws)
  assert len(Ws) == len(bs), "len(Ws) != len(bs)"
  assert num_layer == 2, "currently only support 2-layer NN"
  # assert x0.shape[1] == 1, "currently only support num_batch = 1 computation"
  num_samples = x0.shape[1]

  UBs, LBs = [], []
  if type(eps) is torch.Tensor:
    eps = eps.unsqueeze(0).T
  UBs.append(x0+eps) # (dim_in, num_batch) 
  LBs.append(x0-eps) 

  for i in range(num_layer):
    if i == 0:
      W_pos=Ws[i].clone() # Ws[0]: (num_hidden, dim_in)
      W_neg=Ws[i].clone() 
      W_pos[Ws[i]<0]=0
      W_neg[Ws[i]>0]=0
      UB=torch.matmul(W_pos,UBs[i])+torch.matmul(W_neg,LBs[i])+bs[i] # i=0, (num_hidden, num_batch)
      LB=torch.matmul(W_pos,LBs[i])+torch.matmul(W_neg,UBs[i])+bs[i] 
    else:
      W_pos=Ws[i].clone() # Ws[1]: (dim_out, num_hidden)
      W_neg=Ws[i].clone() 
      W_pos[Ws[i]<0]=0 
      W_neg[Ws[i]>0]=0

      A_U, B_U = get_upper_coefficients(UBs[-1],LBs[-1],activation) # A_U: (num_batch, num_hidden, num_hidden)
      A_L, B_L = get_lower_coefficients(UBs[-1],LBs[-1],activation,method) 
      B_U = B_U.T.unsqueeze(2)  # B_U: (num_batch, num_hidden, 1)
      B_L = B_L.T.unsqueeze(2)
      W_pos = W_pos.repeat(num_samples, 1, 1)
      W_neg = W_neg.repeat(num_samples, 1, 1)

      A_U_new = torch.bmm(W_pos,A_U) + torch.bmm(W_neg,A_L) # shape: (num_batch, dim_out, num_hidden)
      B_U_new = torch.bmm(W_pos,B_U) + torch.bmm(W_neg,B_L) + bs[i] # shape: (num_batch, dim_out, 1)
      A_L_new = torch.bmm(W_pos,A_L) + torch.bmm(W_neg,A_U)
      B_L_new = torch.bmm(W_pos,B_L) + torch.bmm(W_neg,B_U) + bs[i]

      # for 2-layer UB:
      W_equiv_U = torch.matmul(A_U_new, Ws[i-1]) # shape: (num_batch, dim_out, dim_in)
      B_equiv_U = torch.matmul(A_U_new, bs[i-1]) + B_U_new # shape: (num_batch, dim_out, 1)
      W_pos = W_equiv_U.clone()
      W_neg = W_equiv_U.clone()
      W_pos[W_equiv_U<0]=0
      W_neg[W_equiv_U>0]=0
      UB=torch.bmm(W_pos,UBs[0].T.unsqueeze(2))+torch.bmm(W_neg,LBs[0].T.unsqueeze(2))+B_equiv_U 
      UB = UB.squeeze(dim=-1).T # shape: (dim_out, num_batch)

      # for 2-layer LB:
      W_equiv_L = torch.matmul(A_L_new, Ws[i-1])
      B_equiv_L = torch.matmul(A_L_new, bs[i-1]) + B_L_new
      W_pos = W_equiv_L.clone()
      W_neg = W_equiv_L.clone()
      W_pos[W_equiv_L<0]=0
      W_neg[W_equiv_L>0]=0
      LB=torch.bmm(W_pos,LBs[0].T.unsqueeze(2))+torch.bmm(W_neg,UBs[0].T.unsqueeze(2))+B_equiv_L
      LB = LB.squeeze(dim=-1).T # shape: (dim_out, num_batch)

    UBs.append(UB)
    LBs.append(LB)
  
  return UBs, LBs 

def get_barrier_bounds_fastlin_batch_mult_layers(x0, eps, Ws, bs, method="fastlin", activation="relu"):
  # x0: (dim_in, num_batch)
  # initial setting
  # act = lambda x: torch.relu(x) 
  num_layer = len(Ws)
  assert len(Ws) == len(bs), "len(Ws) != len(bs)"
  # assert num_layer == 2, "currently only support 2-layer NN"
  # assert x0.shape[1] == 1, "currently only support num_batch = 1 computation"
  num_samples = x0.shape[1]

  A_Us, A_Ls, B_Us, B_Ls = [], [], [], []

  UBs, LBs = [], []
  if type(eps) is torch.Tensor:
    eps = eps.unsqueeze(0).T
  UBs.append(x0+eps) # (dim_in, num_batch) 
  LBs.append(x0-eps) 

  for i in range(num_layer):
    if i == 0:
      W_pos=Ws[i].clone() # Ws[0]: (num_hidden, dim_in)
      W_neg=Ws[i].clone() 
      W_pos[Ws[i]<0]=0
      W_neg[Ws[i]>0]=0
      
      UB=torch.matmul(W_pos,UBs[i])+torch.matmul(W_neg,LBs[i])+bs[i] # i=0, (num_hidden, num_batch)
      LB=torch.matmul(W_pos,LBs[i])+torch.matmul(W_neg,UBs[i])+bs[i] 
      UBs.append(UB)
      LBs.append(LB)
      
      A_U, B_U = get_upper_coefficients(UBs[-1],LBs[-1],activation) # A_U: (num_batch, num_hidden, num_hidden)
      A_L, B_L = get_lower_coefficients(UBs[-1],LBs[-1],activation,method) 
      B_U = B_U.T.unsqueeze(2)  # B_U: (num_batch, num_hidden, 1)
      B_L = B_L.T.unsqueeze(2)
      A_Us.append(A_U)
      B_Us.append(B_U)
      A_Ls.append(A_L)
      B_Ls.append(B_L)
    else:
      A_eq_U = torch.eye(Ws[i].size()[0])           
      A_eq_U = A_eq_U.repeat(num_samples, 1, 1) # A_eq_U: (num_batch, num_hidden, num_hidden)
      B_eq_U = torch.zeros(num_samples, Ws[i].size()[0], 1)            # B_eq_U: (num_batch, num_hidden, 1)
      A_eq_L = torch.eye(Ws[i].size()[0])           
      A_eq_L = A_eq_L.repeat(num_samples, 1, 1) # A_eq_U: (num_batch, num_hidden, num_hidden)
      B_eq_L = torch.zeros(num_samples, Ws[i].size()[0], 1)            # B_eq_U: (num_batch, num_hidden, 1)

      for j in range(i, -1, -1):
        W_new = Ws[j].clone()
        W_new = Ws[j].repeat(num_samples, 1, 1)
        W_new = torch.bmm(A_eq_U, W_new)

        b_new = bs[j].clone()
        b_new = b_new.repeat(num_samples, 1, 1)
        b_new_U = torch.bmm(A_eq_U, b_new) + B_eq_U
        b_new_L = torch.bmm(A_eq_L, b_new) + B_eq_L

        W_new_pos = W_new.clone()
        W_new_neg = W_new.clone()
        W_new_pos[W_new<0] = 0
        W_new_neg[W_new>0] = 0

        if j == 0:
          break
        A_U = A_Us[j - 1]
        B_U = B_Us[j - 1]
        A_L = A_Ls[j - 1]
        B_L = B_Ls[j - 1]

        A_eq_U = torch.bmm(W_new_pos, A_U) + torch.bmm(W_new_neg, A_L)
        B_eq_U = torch.bmm(W_new_pos, B_U) + torch.bmm(W_new_neg, B_L) + b_new_U
        A_eq_L = torch.bmm(W_new_pos, A_L) + torch.bmm(W_new_neg, A_U)
        B_eq_L = torch.bmm(W_new_pos, B_L) + torch.bmm(W_new_neg, B_U) + b_new_L

      UB=torch.bmm(W_new_pos,UBs[0].T.unsqueeze(2))+torch.bmm(W_new_neg,LBs[0].T.unsqueeze(2))+b_new_U 
      UB = UB.squeeze(dim=-1).T # shape: (dim_out, num_batch)

      LB=torch.bmm(W_new_pos,LBs[0].T.unsqueeze(2))+torch.bmm(W_new_neg,UBs[0].T.unsqueeze(2))+b_new_L
      LB = LB.squeeze(dim=-1).T # shape: (dim_out, num_batch)

      UBs.append(UB)
      LBs.append(LB)

      A_U, B_U = get_upper_coefficients(UBs[-1],LBs[-1],activation) # A_U: (num_batch, num_hidden, num_hidden)
      A_L, B_L = get_lower_coefficients(UBs[-1],LBs[-1],activation,method) 
      B_U = B_U.T.unsqueeze(2)  # B_U: (num_batch, num_hidden, 1)
      B_L = B_L.T.unsqueeze(2)

      A_Us.append(A_U)
      B_Us.append(B_U)
      A_Ls.append(A_L)
      B_Ls.append(B_L)

  return UBs, LBs 


"""## Check no violation"""

def generate_random_sample(x0, eps):
  # input: [dim, batch]
  # rand: [0,1) --> make input range: (-1, 1)
  x_samples = eps*(2*torch.rand(x0.shape)-1)+x0  # x_sample: shape (dim_in, num_batch)
  return x_samples

def show(x,bnds_x):
  plt.figure()
  plt.plot(x,np.zeros(x.shape),marker='o')
  plt.plot(bnds_x,np.zeros(bnds_x.shape),"ro")
  # plt.xlim(0.1,0.3)
  # plt.ylim(0,0.2)
  plt.show()

def validate_bounds(x0, eps, Ws, bs, method, activation="relu", plot=True):
  # x0: [dim, batch]
  # get bounds
  if method == "ibp":
    UBs, LBs = get_barrier_bounds_ibp(x0, eps, Ws, bs, activation) # shape: (dim_out, batch)
  elif method in ["fastlin","crown","crownzero"]: # haven't implemented batch version
    num_samp = x0.shape[1]
    UBs, LBs = [], []
    for i in range(num_samp):
      UBs_tmp, LBs_tmp = get_barrier_bounds_fastlin(x0[:,i:i+1], eps, Ws, bs, method, activation) # shape: (dim_out, batch=1)
      UBs.append(UBs_tmp[-1])
      LBs.append(LBs_tmp[-1])
  else:
    raise ValueError("unknown method!")
  # generate MC samples
  x_samples = generate_random_sample(x0, eps)
  # input size to model: (Batch, dim_in), x0: (dim_in, Batch)
  out_samples = model.forward(x_samples.T).data.T # shape: (dim_out, batch) 
   
  bounds = torch.cat((LBs[-1][:,0],UBs[-1][:,0]),0) # shape: 2
  # print("bounds = {}".format(bounds)) 
  # print(out_samples[0,:].shape, bounds.shape)
  if plot:
    show(out_samples[0,:].numpy(),bounds.numpy())

  # check violations
  violation_UBs = torch.where(UBs[-1] < out_samples)
  violation_LBs = torch.where(LBs[-1] > out_samples)

  for i in range(len(violation_UBs)):
    assert torch.sum(violation_UBs[i]) == 0, "violating UBs[{}]".format(i)  
  print("pass validation of UB!")

  for i in range(len(violation_LBs)):
    assert torch.sum(violation_LBs[i]) == 0, "violating LBs[{}]".format(i)
  print("pass validation of LB!") 

  return violation_UBs, violation_LBs


if __name__ == "__main__":

  dim_in, dim_out, num_hidden, num_batch = 3, 1, 128, 1
  model = nn.Sequential(nn.Linear(dim_in,num_hidden),nn.Tanh(),nn.Linear(num_hidden,dim_out))
  Ws, bs = get_weights(model)

  """## Testing - tanh"""
  # input: [dim, batch]
  # x0 = torch.ones([dim_in, num_batch], dtype=torch.float32)
  num_sample = 1
  x0 = torch.rand(dim_in,1)
  x0 = x0.repeat(1,num_sample) # this is just for plot. To enable batch dimension for ibp bounds, x0 shape: (dim, batch)

  # perturbation magnitude
  eps = 1

  """### IBP bounds"""
  UBs, LBs = get_barrier_bounds_ibp(x0, eps, Ws, bs, activation="tanh")
  print("IBP bounds: UB = {}, LB = {}".format(UBs[-1], LBs[-1]))

  """### Crown bounds"""
  UBs_C, LBs_C = get_barrier_bounds_fastlin(x0, eps, Ws, bs, method="crown", activation="tanh")
  print("CROWN bounds: UB = {}, LB = {}".format(UBs_C[-1], LBs_C[-1]))

  #validate_bounds(x0, eps=0.01, Ws=Ws, bs=bs, method="ibp", activation="tanh")
  #validate_bounds(x0, eps=0.01, Ws=Ws, bs=bs, method="crown", activation="tanh")

  """## Sweep: checking no violations"""
  num_test = 0
  if num_test:
    num_sample = 100
    for k in range(num_test):
      x0 = torch.rand(dim_in,1)
      x0 = x0.repeat(1,num_sample)
      for eps in [0.001, 0.01, 0.1, 0.5, 1]:
        for method in ["crown"]:
          print("===== test {}, eps = {}, method = {} ======".format(k, eps, method))
          vu, vl = validate_bounds(x0, eps, Ws=Ws, bs=bs, method=method, activation="tanh", plot=False)

