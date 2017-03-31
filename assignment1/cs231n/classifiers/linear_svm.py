import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """

  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1] # C
  num_train = X.shape[0]   # D
  loss = 0.0

  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]: #correct class check
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW [:,j] += X[i,:] # compute loss for j class (X matrix CxD)
        dW [:,y[i]] -= X[i,:]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  num_train = X.shape[0]
    
  allrows = np.arange(num_train)
  S = np.dot(X, W)
  scores = S - S[allrows, y][None].T + 1

  margins = np.maximum(0, scores)
  margins[allrows, y] = 0

  loss = 1.0 / num_train * np.sum(margins) + 0.5 * reg * np.sum(W * W)

  margin_ind = np.where(scores > 0, 1, 0)
  margin_ind[allrows, y] = 0
  margin_ind[allrows, y] = -np.sum(margin_ind, axis=1)

  dW = 1.0 / num_train * np.dot(X.T, margin_ind) + reg * W

  return loss, dW
