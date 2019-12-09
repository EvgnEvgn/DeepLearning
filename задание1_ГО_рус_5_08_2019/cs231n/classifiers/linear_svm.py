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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    count = 0
    for j in range(num_classes):
      
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if j == y[i]:
        margin = 0
        
      if margin > 0:
        if j != y[i]:
            count += 1
            dW[:, j] += X[i, :].T
        loss += margin
        
    dW[:, y[i]] += (-count)*(X[i, :].T)
         
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * 2 * W
  #############################################################################
  # ЗАДАНИЕ:                                                                  #
  # Вычислите градиент функции потерь и сохраните его в dW .                  #
  # Вместо того, чтобы сначала вычислять функцию потерь, а затем вычислять    #
  # производную, лучше вычислять производную в процессе вычисления            #
  # функции потерь. Поэтому Вам нужно модифицировать код выше, включив   него #
  # вычисление градиента                                                      #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # ЗАДАНИЕ:                                                                  #
  # Реализуйте векторизованную версию кода для вычисления SVM функции потерь. #
  # Сохраните результат в переменной loss.                                    #
  #############################################################################
  
  num_train = len(y)
 
  scores = X.dot(W)
  correct = scores[range(num_train), y].reshape((num_train, 1))
  margins = np.maximum(0.0, scores - correct + 1.0)
  margins[range(num_train), y] = 0
  loss = margins.sum() / num_train
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             КОНЕЦ ВАШЕГО КОДА                             #
  #############################################################################


  #############################################################################
  # ЗАДАНИЕ:                                                                  #  
  # Реализуйте векторизованную версию кода для вычисления градиента SVM       #
  # функции потерь. Сохраните результат в переменной dW.                      #
  # Совет: Вместо вычисления градиента от начала до конца, лучше использовать #
  # некоторые промежуточные значения, которые были получены при вычислении    #
  # функции потерь.                                                           #
  #############################################################################
  
  dscores = margins
  dscores[margins > 0] = 1
  dscores[range(num_train), y] = -(np.sum(dscores, axis=1)).T
 
  dscores /= X.shape[0]
  dreg = reg * W
  dW = X.T.dot(dscores) + dreg
  #############################################################################
  #                             КОНЕЦ ВАШЕГО КОДА                             #
  #############################################################################

  return loss, dW
