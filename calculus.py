import math


def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def sigmoid_grad(z):
    return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
    return max(z, 0.0)

def relu_grad(z):
    return 1.0 if z > 0.0 else 0.0


if __name__ == '__main__':
    print('# Calculus Gradient #')
    a, b, c, x, y, d = 1, 2, 3, 4, 5, 6

    # Max activation
    x = max(a, b)
    print('Function: max(a[{}], b[{}])={}'.format(a, b, x))
    dx = 1.0  # init grad
    if x == a:
        da = dx * 1.0
        db = 0.0
    else:
        da = 0.0
        db = dx * 1.0
    print('Gradient: da[{}], db[{}]'.format(da, db))
    
    # ReLU activation
    x = relu(a)
    print('Function: relu(a[{}], 0.0)={}'.format(a, x))
    dx = 1.0
    da = dx * relu_grad(a)
    print('Gradient: da[{}]'.format(da))

    # sigmoid(a*x + b*y + c)
    m = a * x
    n = b * y
    q = m + n + c
    f = sigmoid(q)
    print('Function: sigmoid(a[{}]*x[{}] +b[{}]*y[{}] + c[{}])={}'.format(
        a, x, b, y, c, f
    ))
    df = 1.0  # init grad
    dq = df * sigmoid_grad(q)
    dm = dq * 1.0
    dn = dq * 1.0
    dc = dq * 1.0
    da = dm * x
    dx = dm * a
    db = dn * y
    dy = dn * b
    print('Gradient: da[{}], db[{}], dc[{}], dx[{}], dy[{}]'.format(
        da, db, dc, dx, dy
    ))
    
    # (a+b)/(c+d)
    def m(a, b, c, d):
        return (a + b) / (c + d)
    f = m(a, b, c, d)
    print('Function: (a[{}]+b[{}])/(c[{}]+d[{}])={}'.format(
        a, b, c, d, f
    ))
    x1 = a + b
    x2 = c + d
    x3 = 1 / x2
    # f = x1 * x3
    df = 1.0  # init grad
    dx1 = df * x3
    da = dx1 * 1.0
    db = dx1 * 1.0
    dx3 = df * x1
    dx2 = dx3 * (-1/(x2**2))
    dc = dx2 * 1.0
    dd = dx2 * 1.0
    print('Calculus Gradient')
    print('da[{}], db[{}], dc[{}], dd[{}]'.format(
        da, db, dc, dd
    ))
    h = 0.001
    print('Numerical Gradient')
    print('da[{}], db[{}], dc[{}], dd[{}]'.format(
        (m(a+h, b, c, d)-m(a, b, c, d))/h,
        (m(a, b+h, c, d)-m(a, b, c, d))/h,
        (m(a, b, c+h, d)-m(a, b, c, d))/h,
        (m(a, b, c, d+h)-m(a, b, c, d))/h,
    ))
