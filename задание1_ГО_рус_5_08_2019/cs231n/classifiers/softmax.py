import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax фнкция потерь, наивная реализация (с циклами)

  Число пикселей изображения - D, число классов - С, мы оперируем миниблоками по N примеров
  

  Входы:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Возвращает кортеж:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Инициализация потерь и градиентов.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # ЗАДАНИЕ: Вычислите softmax  потери и  градиенты, используя явные циклы.   #
  # Сохраните потери в переменной loss, а градиенты в dW.  Не забывайте о     #
  # регуляризации!                                                            #
  #############################################################################
  
  num_train = y.shape[0]
  num_classes = W.shape[1]
  for i in range(num_train):
        
      scores = X[i].dot(W)
      
      scores -= np.max(scores)
      exps = np.exp(scores)
      exps_sum = np.sum(exps)
      
      p = exps/exps_sum
      L_i = -np.log(p[y[i]])
      
      loss += L_i
      for j in range(num_classes):
        dW[:, j] += (p[j] - (j == y[i])) * X[i, :]
  
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          КОНЕЦ ВАШЕГО КОДА                                #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax функция потерь, векторизованная версия.

  Входы и выходы те же, что и у функции softmax_loss_naive.
  """
  # Инициализация потерь и градиентов.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # ЗАДАНИЕ: Вычислите softmax  потери и  градиенты без использования циклов. #
  # Сохраните потери в переменной loss, а градиенты в dW.  Не забывайте о     #
  # регуляризации!                                                            #
  #############################################################################
  num_train = len(y)
  scores = X.dot(W)
  scores -= np.max(scores)
  exps = np.exp(scores)
  p = exps / np.sum(exps, axis =1, keepdims = True)
  loss = np.sum(-np.log(p[range(num_train), y])) / num_train
  loss += 0.5 * reg * np.sum(W * W)
    
  dscores = p
  dscores[range(num_train), y] -= 1.0
  dreg = reg * W
  dW = X.T.dot(dscores) / num_train + dreg
  #############################################################################
  #                         КОНЕЦ ВАШЕГО КОДА                                 #
  #############################################################################

  return loss, dW

