from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]  # В машинном обучении градиентный спуск считается как среднее градиентов на обучающей выборке. Так, потому что лосс функция - средняя по лоссам на выборке
                dW[:, y[i]] -= X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W # т.к. ф-ция лосса получает еще регуляризацию, то мы ее учитываем и в градиенте
    # (у нас регуляризация L2, т.е. reg * W ** 2) Производная, соответственно, 2 * reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    score_matrix = X.dot(W) # NxC matrix
    # score formula = sum_j sum_i i!=j max(0, si - sj + 1)
    sj_scores = score_matrix[:, y].diagonal() # скоры правильных классов
    score_matrix = score_matrix - sj_scores[:, np.newaxis] + 1 # Вычитаем из каждой строчки скор ее класса и прибавляем 1 ко всей строке
    mask = score_matrix <= 0 # Создаем маску Хинж лосса
    score_matrix[mask] = 0 # Применяем ее к каждому элементу
    mask_function = np.vectorize(lambda x: 0 if x == 0 else 1)
    mask_matrix = mask_function(x)
    score_matrix = score_matrix.sum(axis=1) # Суммируем лоссы на объектах
    score_matrix -= 1 # Вычитаем из сумм по 1. Т.к. в нашем решении мы не учитывали i!=j, то лосс на каждом объекте, когда считалось max(0, si - sj + 1) (i = j) увеличился на 1.
    loss += score_matrix.sum() / len(score_matrix) # Считаем лосс как среднее по лоссам на объектах

    loss += reg * np.sum(W * W) # Добавим регуляризацию

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    mask_matrix[np.arange(mask_matrix.shape[0]), y] -= mask_matrix.sum(axis=1)
    dx = mask_matrix
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
