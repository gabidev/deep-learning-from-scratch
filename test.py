import numpy as np
import matplotlib.pylab as plt

# 활성화함수: 시그모이드
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 활성화함수: 소프트맥스
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

# 손실함수: 평균 제곱 오차
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

# 손실함수: 교차 엔트로피 오차
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

# 미분: 중심 차분(중앙 차분). 단일 x 값에 대한 계산.
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h))/ (2 * h)

# 기울기 계산(여러 x 값에 대한 미분 계산)
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
    
    return grad

# 경사하강법(numerical_gradient로 구한 기울기를 lr 값 만큼 변경하여 다시 계산하는 방식)
def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
        print("i: {0}, grad: {1}, x: {2}".format(i, grad, x))
    
    return x

def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])

print(gradient_descent(function_2, init_x = init_x, lr = 0.1, step_num = 100))

#print(numerical_gradient(function_2, np.array([3.0, 4.0])))
#print(numerical_gradient(function_2, np.array([0.0, 2.0])))
#print(numerical_gradient(function_2, np.array([3.0, 0.0])))

#X = [-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
#Y = [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]
#XX = np.array([-4.000000000132786, -3.5000000002582965, -3.0000000000995897, -2.499999999940883, -2.000000000066393, -1.4999999999076863, -0.9999999997489795, -0.49999999987448973, 0.0, 0.49999999987448973, 0.9999999997489795, 1.4999999999076863, 2.000000000066393, 2.499999999940883])
#YY = np.array([-4.000000000132786, -4.000000000132786, -4.000000000132786, -4.000000000132786, -4.000000000132786, -4.000000000132786, -4.000000000132786, -4.000000000132786, -4.000000000132786, -4.000000000132786, -4.000000000132786, -4.000000000132786, -4.000000000132786, -4.000000000132786])

#X = [1.25]
#Y = [-2.0]
#XX = [2.499999999940883]
#YY = [-4.000000000132786]

#plt.figure()
#plt.quiver(X, Y, -XX, -YY)
#plt.xlim([-3, 3])
#plt.ylim([-3, 3])
#plt.xlabel('x0')
#plt.ylabel('x1')
#plt.grid()
#plt.legend()
#plt.draw()
#plt.show()
