from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(object):
  """
  
  Двухслойная полносвязанная нейронная сеть. Размерность входа сети N, размерность
  скрытого слоя H, сеть выполняет классификацию по C классам.
  Матрица весов сети обучается с использованием  softmaх функции потерь  и L2 
  регуляризации. На выходе первого полносвязного слоя используется нелинейность
  ReLU.

  Другими словами, архитектура сети следующая:

  вход - полносвязный слой - ReLU - полносвязный слой - softmax

  Выходы второго полносвязного слоя -- рейтинги (scores) классов
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-7):
    """
    Инициализирует модель. Веса инициализируются малыми случайными значениями,
    а смещения - нулями. Веса и смещения сохраняются в переменной  self.params, 
    которая представляет собой словарь со следующими ключами:

    W1: Веса первого слоя; размерность (D, H)
    b1: Смещения первого слоя; размерность (H,)
    W2: Веса второго слоя; размерность(H, C)
    b2: Смещения вотрого слоя; размерность (C,)

    Входы:
    - input_size: Размерность D входных данных.
    - hidden_size: Число нейронов скрытого слоя - H.
    - output_size: Число классов - C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Вычисляет функцию потерь и её градиенты для 2-х слойной полносвязной 
    нейронной сети

    Входы:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Вовращает:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Излечение переменных из словаря параметров
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Вычисления прямого распространения
    scores = None
    #############################################################################
    # ЗАДАНИЕ: Выполнить прямое распостранение, вычислив рейтинги классов.      #
    # Сохранить результаты в переменной scores, которая должна быть массивом    #
    # размерности (N, C).                                                       #
    #############################################################################
    #layer1 = X.dot(W1) + b1
    #a1 = np.maximum(layer1, 0.0)
    hidden_layer = np.maximum(0, np.dot(X, W1) + b1)
    scores = np.dot(hidden_layer, W2) + b2
    #############################################################################
    #                              КОНЕЦ ВАШЕГО КОДА                            #
    #############################################################################
    
    # Если метки не заданы, то выход
    if y is None:
      return scores

    # Вычисление функции потерь
    loss = None
    #############################################################################
    # ЗАДАНИЕ: Завершите прямое рапространение и вычислите потери. Они должны   #
    # включать как потери на данных, так и потери L2 регуляризации для W1 и W2. #
    # Сохраните результат в переменной loss, которая должна быть скаляром.       #
    # Используйте функцию потерь Softmax классификатоа                           #
    #############################################################################
    
    scores -= np.max(scores)
    exps = np.exp(scores)
    p = exps / exps.sum(axis =1, keepdims = True)
    loss = -np.log(p[range(N), y]).sum() / N
    loss += 0.5 * reg * np.sum(W1 * W1) +  0.5 * reg * np.sum(W2 * W2)
    
    #############################################################################
    #                              КОНЕЦ ВАШЕГО КОДА                            #
    #############################################################################

    # Обратное рапространение: вычисление градиентов
    grads = {}
    #############################################################################
    # ЗАДАНИИЕ: Выполните обратное рапространение, вычислив производные по весам#
    # и смещениям. Сохраните результаты в словаре grads. Например,              #
    # grads['W1'] хранит градиент по W1, и является матрицей такого же размера, #
    #  что и W1.
    #############################################################################
    dscores = p
    dscores[range(N), y] -= 1
    dscores /= N
    
    grads["b2"] = dscores.sum(axis = 0)
    grads["W2"] = np.dot(hidden_layer.T, dscores)
    
    dlayer1 = np.dot(dscores, W2.T)
    dlayer1[hidden_layer <= 0] = 0
    grads["b1"] = dlayer1.sum(axis = 0)
    grads["W1"] = np.dot(X.T, dlayer1)
    
    grads["W1"] += reg * W1
    grads["W2"] += reg * W2
    #############################################################################
    #                              КОНЕЦ ВАШЕГО КОДА                            #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Обучение нейросети на основе стохастического градиентного спуска

    Входы:
    - X: numpy массив размерности (N, D), содержащий обучающие данные.
    - y: numpy массив размерности (N,), содержащий обучающие метки; y[i] = c 
      означает, что X[i] иеет метку c, где 0 <= c < C.
    - X_val: numpy массив размерности (N_val, D), содержащий валидацонные данные.
    - y_val: numpy массив размерности (N_val,), содержащи валидационные метки.
    - learning_rate: скорость обучения, скаляр.
    - learning_rate_decay: скаляр, определяющий затухание скорости обучения на 
      после каждой эпохи.
    - reg: коээфициент регуляризации, скаляр.
    - num_iters: число итераций при оптимизации.
    - batch_size: число обучающих примров для одного шага.
    - verbose: boolean; если истина, то отображает прогресс оптимизации.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Использует SGD для оптимизации параметров в self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # ЗАДАНИЕ: Создайте случайный мини блок обучающих данных  и меток,      #
      # сохраните их в X_batch и y_batch, соответственно.                     #
      #########################################################################
      mask = np.random.choice(num_train, batch_size)
      X_batch = X[mask]
      y_batch = y[mask]
      #########################################################################
      #                             КОНЕЦ ВАШЕГО КОДА                         #
      #########################################################################

      
      # Вычисление потерь и градиентов для текущего миниблока
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # ЗАДАНИЕ: Используйте градиенты в словаре grads для обновления         #
      # параметров сети (запомненных в словаре self.params)                   #
      # в соответствии с алгоритмом SGD.                                      #
      #########################################################################
      for w in self.params:
          self.params[w] -= learning_rate * grads[w]
     
      #########################################################################
      #                            КОНЕЦ ВАШЕГО КОДА                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # На каждой эпохе проверяем точности обучения и валидации (val), 
      # вычисляем затухание скорости обучения
      if it % iterations_per_epoch == 0:
        # Проверка точности
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Затухание скорости обучения
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Использует обученные веса 2-х слойной сети для предсказания меток образцов
    данных. Для каждого образца (примера) данных вычисляются рейтинги
    принадлежности C классам и выполняется связывание с тем классом, который
    имеет максимальный рейтинг 

    Входы:
    - X:  numpy массив размерности (N, D), содержащий N D-мерных образцов данных,
      подлежащих классификации.

    Возвращает:
    - y_pred: numpy массив размерности (N,) содержащий предсказанные метки для 
      каждого элемента X. Для всех i, y_pred[i] = c означает, что X[i] принадлежит
      классу c, где 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # ЗАДАНИЕ: Реализуйте эту функцию. Она должна ыбть очень простой!         #
    ###########################################################################
    layer1 = X.dot(self.params["W1"]) + self.params["b1"]
    a1 = np.maximum(0, layer1)
    scores = a1.dot(self.params["W2"]) + self.params["b2"]
    y_pred = scores.argmax(axis=1)
    ###########################################################################
    #                             КОНЕЦ ВАШЕГО КОДА                           #
    ###########################################################################

    return y_pred


