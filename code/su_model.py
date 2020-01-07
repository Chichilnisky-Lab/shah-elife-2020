"""Jointly fit subunits and output NL."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle
import numpy as np, numpy
import tensorflow as tf
from tensorflow.python.platform import gfile

rng = np.random


def compute_fr_loss(K, b, X_in, Y_in, nl_params=np.expand_dims(np.array([1.0, 0.0]), 1)):
  """ Compute firing rate and loss.

  Args :
    K : Subunit filters, (dims: # n_pix x #SU).
    b : Subunit weights for different cells (dims: # SU x # cells).
    X_in : Stimulus (dims: # samples x # pixels).
    Y_in : Responses (dims: # samples x # cells).
    nl_params : Nonlinearity parameters (dims: 2 x 1, defaults to no nonlinearity).
    
  Returns :
    fsum : Predicted firing rate (dims: # samples x # cells).
    loss : Prediction loss (dims : # cells).
  """
  f = np.exp(np.expand_dims(np.dot(X_in, K), 2) + b)  # T x SU x Cells
  fsum = f.sum(1)  # T x # cells
  # apply nonlinearity
  fsum = np.power(fsum, nl_params[0, :]) / (nl_params[1, :] * fsum + 1)
  loss = np.mean(fsum, 0) - np.mean(Y_in * np.log(fsum), 0)  # cells
  return fsum, loss


def get_neighbormat(mask_matrix, nbd=1):
  """ Compute the adjacency matrix for nearby pixels
  
  Args: 
    mask_matrix : 2D visual stimulus mask, active pixels are 1 (dims : # stimulus dim 1 x # stimulus dim 2).
    nbd : Neighborhood size (scalar).
    
  Returns : 
    neighbor_mat : 2D adjacency matrix (dims: # active pixels x # active pixels).
  """
  mask = np.ndarray.flatten(mask_matrix)>0

  x = np.repeat(np.expand_dims(np.arange(mask_matrix.shape[0]), 1), mask_matrix.shape[1], 1)
  y = np.repeat(np.expand_dims(np.arange(mask_matrix.shape[1]), 0), mask_matrix.shape[0], 0)
  x = np.ndarray.flatten(x)
  y = np.ndarray.flatten(y)

  idx = np.arange(len(mask))
  iidx = idx[mask]
  xx = np.expand_dims(x[mask], 1)
  yy = np.expand_dims(y[mask], 1)

  distance = (xx - xx.T)**2 + (yy - yy.T)**2
  neighbor_mat = np.double(distance <= nbd)

  return neighbor_mat


def spike_triggered_clustering(X, Y, Ns, tms_tr, tms_tst, K=None, b=None,
                               steps_max=10000, eps=1e-6,
                               projection_type=None, neighbor_mat=None,
                               lam_proj=0, eps_proj=0.01,
                               save_filename_partial=None,
                               fitting_phases=[1, 2, 3]):
  """ Subunit estimation using spike triggered clustering.
  
  The fitting proceeds in three phases - 
  First phase: Ignoring the output nonlinearity and soft-clustering of spike triggered stimuli to estimate K and b.
  Second phase: Fix K, optimize b and output nonlinearity by gradient descent.
  Third phase : Optimize K, b and the nonlinearity by gradient descent.
  
  Args: 
    X : Stimulus (dims: # samples x # pixels).
    Y: Responses (dims: # samples x # cells).
    Ns: Number of subunits (scalar).
    tms_tr: Sample indices used for training (dims: # training samples).
    tms_tst: Samples indices for validation (dims: # validation samples).
    K: Initial subunit filter (dims: # pixels x Ns).
    b: Initial weights for different subunits (dims: Ns x # cells).
    steps_max: Maximum number of steps for first phase.
    eps: Threshold change of loss, for convergence.
    projection_type: Regularization type ('lnl1' or 'l1').
    neighbor_mat: Adjacency matrix for pixels (dims: # pixels x # pixels).
    lam_proj: Regularization strength.
    eps_proj: Hyperparameter for 'lnl1' regularization.
    save_filename_partial: Checkpoint filename.
    fitting_phases: Phases of fitting to be applied (list with elements from {1, 2, 3}).
    
  Returns:
    K : Final subunit filters, (dims: # n_pix x #SU).
    b : Final subunit weights for different cells (dims: # SU x # cells).
    alpha: Softmax weights for each stimulus and different subunits (dims: # samples x Ns).
    lam_log: Training loss curve.
    lam_log_test: Validation loss curve.
    fitting_phase: The phase (1/2/3) corresponding to each iteration.
    fit_params: Outputs (K, b, nonlinearity parameters) after each fitting phase. 
  """
  # projection_op='lnl1'

  # X is Txmask
  X_tr = X[tms_tr, :]
  Y_tr = Y[tms_tr, :]
  X_test = X[tms_tst, :]
  Y_test = Y[tms_tst, :]

  Tlen = Y_tr.shape[0]
  times = np.arange(Tlen)
  N1 = X_tr.shape[1]
  n_cells = Y.shape[1]
  Sigma = numpy.dot(X_tr.transpose(),X_tr)/float(X_tr.shape[0])
  
  if projection_type == 'lnl1':
    if neighbor_mat is None:
      neighbor_mat = np.eye(N1)


  # load previously saved data
  if gfile.Exists(save_filename_partial):
    try:
      data = pickle.load(gfile.Open(save_filename_partial, 'r'))
      K = data['K']
      b = data['b']
      lam_log = data['lam_log']
      lam_log_test = data['lam_log_test']
      irepeat_start = data['irepeat']

      lam = lam_log[-1]
      lam_test = lam_log_test[-1]
      lam_min = data['lam_min']
      K_min = data['K_min']
      b_min = data['b_min']
      #print('Partially fit model parameters loaded')
    except:
      pass
      #print('Error in loading file')

      if K is None:
        K = 2*rng.rand(N1,Ns)-0.5
      K_min = np.copy(K)

      if b is None:
        b = 2*rng.rand(Ns, n_cells)-0.5
      b_min = np.copy(b)

      lam_log = np.zeros((0, n_cells))
      lam_log_test = np.zeros((0, n_cells))
      lam = np.inf
      lam_test = np.inf
      lam_min = np.inf
      irepeat_start = 0

  else:
    #print('No partially fit model')
    # initialize filters
    if K is None:
      K = 2*rng.rand(N1,Ns)-0.5
    K_min = np.copy(K)

    if b is None:
      b = 2*rng.rand(Ns, n_cells)-0.5
    b_min = np.copy(b)

    lam_log = np.zeros((0, n_cells))
    lam_log_test = np.zeros((0, n_cells))
    lam = np.inf
    lam_test = np.inf
    lam_min = np.inf
    irepeat_start = 0
    #print('Variables initialized')

  fitting_phase = np.array([])
  fit_params = []

  # Find subunits - no output NL
  if 1 in fitting_phases:
    for irepeat in range(irepeat_start, np.int(steps_max)):

      if irepeat % 100 == 99:
        save_dict = {'K': K, 'b': b, 'lam_log': lam_log,
                     'lam_log_test': lam_log_test, 'irepeat': irepeat,
                     'K_min': K_min, 'b_min': b_min, 'lam_min': lam_min}
        if save_filename_partial is not None:
          pickle.dump(save_dict, gfile.Open(save_filename_partial, 'w' ))

      # compute reweighted L1 weights
      if projection_type == 'lnl1':
        wts = 1 / (neighbor_mat.dot(np.abs(K)) + eps_proj)

      # test data
      _, lam_test = compute_fr_loss(K, b, X_test, Y_test)
      lam_log_test = np.append(lam_log_test, np.expand_dims(lam_test, 0), 0)

      # train data
      lam_prev = np.copy(lam)
      _, lam = compute_fr_loss(K, b, X_tr, Y_tr)
      lam_log = np.append(lam_log, np.expand_dims(lam, 0), 0)

      if np.sum(lam) <= np.sum(lam_min) :
        K_min = np.copy(K)
        b_min = np.copy(b)
        lam_min = np.copy(lam)
        lam_test_at_lam_min = np.copy(lam_test)

      #print(itime)
      K_new_list_nr = []
      K_new_list_dr = []
      mean_ass_f_list = []
      for icell in range(n_cells):
        tms = np.int64(np.arange(Tlen))
        t_sp = tms[Y_tr[:, icell] != 0]
        Y_tsp = Y_tr[t_sp, icell]

        f = np.exp(numpy.dot(X_tr, K) + b[:, icell])
        alpha = (f.transpose()/f.sum(1)).transpose()
        xx = (Y_tsp.transpose()*alpha[t_sp, :].T).T
        sta_f = X_tr[t_sp,:].transpose().dot(xx)
        mean_ass_f = xx.sum(0)

        K_new_list_nr += [numpy.linalg.solve(Sigma,sta_f)]
        K_new_list_dr += [mean_ass_f]
        mean_ass_f_list += [mean_ass_f]

      K_new_list_nr = np.array(K_new_list_nr)
      K_new_list_dr = np.array(K_new_list_dr)
      mean_ass_f_list = np.array(mean_ass_f_list).T # recompute ??

      K = np.mean(K_new_list_nr, 0) / np.mean(K_new_list_dr, 0)

      # Soft thresholding for K
      if projection_type == 'lnl1':
        K = np.maximum(K - (wts * lam_proj), 0) - np.maximum(- K - (wts * lam_proj), 0)

      if projection_type == 'l1':
        K = np.maximum(K - lam_proj, 0) - np.maximum(- K - lam_proj, 0)

      b = np.log((1/Tlen)*mean_ass_f_list)- np.expand_dims(np.diag(0.5*K.transpose().dot(Sigma.dot(K))), 1)

      #print(irepeat, lam, lam_prev)

      if np.sum(np.abs(lam_prev - lam)) < eps:
        print('Subunits fitted, Train loss: %.7f, '
              'Test loss: %.7f after %d iterations' % (lam, lam_test, irepeat))
        break

    fitting_phase = np.append(fitting_phase, np.ones(lam_log.shape[0]))
    nl_params = np.repeat(np.expand_dims(np.array([1.0, 0.0]), 1), n_cells, 1)
    fit_params += [[np.copy(K_min), np.copy(b_min), nl_params]]

  # fit NL + b + Kscale
  if 2 in fitting_phases:
    K, b, nl_params, loss_log, loss_log_test = fit_scales(X_tr, Y_tr,
                                                          X_test, Y_test,
                                                          Ns=Ns, K=K, b=b,
                                                          params=nl_params,
                                                          lr=0.001, eps=eps)

    if 'lam_log' in vars():
      lam_log = np.append(lam_log, np.array(loss_log), 0)
    else:
      lam_log = np.array(loss_log)

    if 'lam_log_test' in vars():
      lam_log_test = np.append(lam_log_test, np.array(loss_log_test), 0)
    else:
      lam_log_test = np.array(loss_log_test)

    fitting_phase = np.append(fitting_phase, 2 * np.ones(np.array(loss_log).shape[0]))
    fit_params += [[np.copy(K), np.copy(b), nl_params]]

  # Fit all params
  if 3 in fitting_phases:
    K, b, nl_params, loss_log, loss_log_test  = fit_all(X_tr, Y_tr, X_test, Y_test,
                                                     Ns=Ns, K=K, b=b, train_phase=3,
                                                     params=nl_params,
                                                     lr=0.001, eps=eps)

    if 'lam_log' in vars():
      lam_log = np.append(lam_log, np.array(loss_log), 0)
    else:
      lam_log = np.array(loss_log)

    if 'lam_log_test' in vars():
      lam_log_test = np.append(lam_log_test, np.array(loss_log_test), 0)
    else:
      lam_log_test = np.array(loss_log_test)

    fitting_phase = np.append(fitting_phase, 3 * np.ones(np.array(loss_log).shape[0]))
    fit_params += [[np.copy(K), np.copy(b), nl_params]]

  return K, b, alpha, lam_log, lam_log_test, fitting_phase, fit_params


def fit_scales(X_tr, Y_tr, X_test, Y_test,
               Ns=5, K=None, b=None, params=None, lr=0.1, eps=1e-9):
  """Second phase of fitting. """
  
  
  X = tf.placeholder(tf.float32)  # T x Nsub
  Y = tf.placeholder(tf.float32)  # T x n_cells

  # initialize filters
  if K is None or b is None or params is None:
      raise "Not initialized"

  K_tf_unscaled = tf.constant(K.astype(np.float32))
  K_scale = tf.Variable(np.ones((1, K.shape[1])).astype(np.float32))

  K_tf = tf.multiply(K_tf_unscaled, K_scale)
  b_tf = tf.Variable(b.astype(np.float32))
  params_tf = tf.Variable(np.array(params).astype(np.float32))  # 2 x # cells

  lam_int = tf.reduce_sum(tf.exp(tf.expand_dims(tf.matmul(X, K_tf), 2) + b_tf), 1)  # T x # cells
  # lam = params_tf[0]*lam_int / (params_tf[1]*lam_int + 1)
  lam = tf.pow(lam_int, params_tf[0, :])/ (params_tf[1, :] * lam_int + 1) # T x # cells
  loss = tf.reduce_mean(lam, 0) - tf.reduce_mean(Y * tf.log(lam), 0)
  loss_all_cells = tf.reduce_sum(loss)

  train_op = tf.train.AdamOptimizer(lr).minimize(loss_all_cells, var_list=[K_scale, b_tf, params_tf])

  with tf.control_dependencies([train_op]):
    #param_pos = tf.assign(params_tf[1], tf.nn.relu(params_tf[1]))
    param_pos = params_tf[1].assign(tf.nn.relu(params_tf[1]))

  train_op_grp = tf.group(train_op, param_pos)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    K_min = sess.run(K_tf)
    b_min = sess.run(b_tf)
    params_min = sess.run(params_tf)
    l_tr_log = []
    l_test_log = []
    l_tr_prev = np.inf
    l_min = np.inf
    for iiter in range(100000):
      l_tr, _ = sess.run([loss, train_op_grp], feed_dict={X: X_tr, Y: Y_tr})
      l_test = sess.run(loss, feed_dict={X: X_test, Y: Y_test})

      l_tr_log += [l_tr]
      l_test_log += [l_test]
      # from IPython import embed; embed()
      # print(iiter, l_tr)
      # print('.', end='')
      if np.sum(l_tr) < np.sum(l_min) :
        K_min = sess.run(K_tf)
        b_min = sess.run(b_tf)
        params_min = sess.run(params_tf)
        l_min = l_tr

      if np.sum(np.abs(l_tr_prev - l_tr)) < eps:
        # print('Nonlinearity fit after : %d iters, Train loss: %.7f' % (iiter, l_tr))
        break
      l_tr_prev = l_tr

    return K_min, b_min, params_min, l_tr_log, l_test_log


def fit_all(X_tr, Y_tr, X_test, Y_test,
            Ns=5, K=None, b=None, params=None,
            train_phase=2, lr=0.1, eps=1e-9):
  """Third phase of fitting. """
  
  X = tf.placeholder(tf.float32)  # T x Nsub
  Y = tf.placeholder(tf.float32)  # T

  # initialize filters
  if K is None or b is None or params is None:
    raise "Not initialized"

  K_tf = tf.Variable(K.astype(np.float32))
  b_tf = tf.Variable(b.astype(np.float32))
  params_tf = tf.Variable(np.array(params).astype(np.float32))

  lam_int = tf.reduce_sum(tf.exp(tf.expand_dims(tf.matmul(X, K_tf), 2) + b_tf), 1) # T x # cells
  # lam = params_tf[0]*lam_int / (params_tf[1]*lam_int + 1)
  lam = tf.pow(lam_int, params_tf[0, :])/ (params_tf[1, :] * lam_int + 1) # T x # cells
  loss = tf.reduce_mean(lam, 0) - tf.reduce_mean(Y * tf.log(lam), 0)
  loss_all_cells = tf.reduce_sum(loss)

  if train_phase == 2:
    train_op = tf.train.AdamOptimizer(lr).minimize(loss_all_cells, var_list=[b_tf, params_tf])
  if train_phase == 3:
    train_op = tf.train.AdamOptimizer(lr).minimize(loss_all_cells, var_list=[K_tf, b_tf, params_tf])

  with tf.control_dependencies([train_op]):
    param_pos = tf.assign(params_tf[1], tf.nn.relu(params_tf[1]))

  train_op_grp = tf.group(train_op, param_pos)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    K_min = sess.run(K_tf)
    b_min = sess.run(b_tf)
    params_min = sess.run(params_tf)

    l_tr_log = []
    l_test_log = []
    l_tr_prev = np.inf
    l_min = np.inf
    for iiter in range(100000):
      l_tr, _ = sess.run([loss, train_op_grp], feed_dict={X: X_tr, Y: Y_tr})
      l_test = sess.run(loss, feed_dict={X: X_test, Y: Y_test})

      l_tr_log += [l_tr]
      l_test_log += [l_test]

      #print(iiter, l_tr)
      if np.sum(l_tr) < np.sum(l_min) :
        K_min = sess.run(K_tf)
        b_min = sess.run(b_tf)
        params_min = sess.run(params_tf)
        l_min = l_tr

      if np.sum(np.abs(l_tr_prev - l_tr)) < eps:
        # print('Nonlinearity fit after : %d iters, Train loss: %.7f' % (iiter, l_tr))
        break
      l_tr_prev = l_tr

    return K_min, b_min, params_min, l_tr_log, l_test_log
