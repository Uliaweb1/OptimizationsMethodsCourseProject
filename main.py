import math
import matplotlib.pyplot as plt

# Допоміжні функції з лінійної алгебри
# Норма вектора
def norm(r):
    return math.sqrt(sum([x * x for x in r]))

# Додавання векторів
def vvadd(A, B):
    if(len(A) == len(B)):
        n = len(A)
        return [A[i] + B[i] for i in range(n)]

# Віднімання векторів
def vvsubtract(A, B):
    if(len(A) == len(B)):
        n = len(A)
        return [A[i] - B[i] for i in range(n)]

# Скалярний добуток векторів
def vvdot(A, B):
    if(len(A) == len(B)):
        n = len(A)
        return sum([A[i] * B[i] for i in range(n)])

# Зовнішній добуток векторів (тобто a на b^T)
def vvouter(A, B):
    rows = len(A)
    return [constv(A[i], B) for i in range(rows)]

# Додавання матриць
def mmadd(A, B):
    if(len(A) == len(B)):
        rows = len(A)
        return [vvadd(A[i], B[i]) for i in range(rows)]

# Транспонування матриці
def mtanspose(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

# Множення матриці на вектор
def mvmult(A, v):
    return [vvdot(A[i], v) for i in range(len(A))]

# Множення матриць
def mmmult(A, B):
    tB = mtanspose(B)
    cols = len(tB)
    # print(f"\tmmmult: A({A}), tB({tB})")
    return mtanspose([mvmult(A, tB[i]) for i in range(cols)])

# Множення вектора на константу
def constv(c, A):
    return [c * A[i] for i in range(len(A))]

# Ділення вектора на константу
def constvdiv(c, A):
    return [c * A[i] for i in range(len(A))]

# Множення матриці на константу
def constm(c, A):
    return [constv(c, A[i]) for i in range(len(A))]

# Ділення матриці на константу
def constmdiv(c, A):
    return [constv(c, A[i]) for i in range(len(A))]

# Обернена 2x2 матриця
def Inverse22(A):
    [[A11, A12], [A21, A22]] = A
    return constmdiv(A11 * A22 - A12 * A21, [[A22, -A21], [-A12, A11]]) # за правелом Крамера

def IsPositiveM(A):
    [[A11, A12], [A21, A22]] = A
    return A11 > 0.0 and A12 > 0.0 and A21 > 0.0 and A22 > 0.0
# Кінець блоку допоміжних функцій з лінійної алгебри

# Алгоритм Свена для пошуку інтервалу
def sven(f, x0, l):
    d =  l
    xm, x, xp = x0 - d, x0, x0 + d
    fm, fx, fp = f(xm), f(x), f(xp)

    if fm > fx and fp > fx:
        return (xm, xp) # Вже в локальному мінімумі
    if fx > fm:
        # Зміна напрямку пошуку
        d = -d
        xm, xp = xp, xm

    while fx > fp:
        d += 2
        xm, x, xp = x, xp, xp + d
        fm, fx, fp = fx, fp, f(xp)
    xmid = (x + xp) / 2
    fmid = f(xmid)

    if fmid > fx:
        (xl, xr) = (xm, xmid)
    else:
        (xl, xr) = (x, xp)

    if xl > xr:
        xl, xr = xr, xl
    return (xl, xr)

# Знаходження інтервалу, на якому є мінімум функції
def find_interval(f, x0, l = 1):
    return sven(f, x0, l)

# Метод бісекції для пошуку локального мінімума на інтервалі
def bisection_method(f, a, b, eps):
    xm = (a + b) / 2
    x1 = a + xm / 2
    x2 = (xm + b) / 2
    fa, f1, fm, f2, fb = f(a), f(x1), f(xm), f(x2), f(b)
    while (b - a) > eps:
        if fm < f1 and fm < f2:
            a, b, fa, fb = x1, x2, f1, f2
        elif f1 > f2:
            a, xm, fa, fm = xm, x2, fm, f2
        else:
            xm, b, fm, fb = x1, xm, f1, fm
        x1, x2 = (a + xm) / 2, (xm + b) / 2
        f1, f2 = f(x1), f(x2)
    return xm

# Пошук локального мінімума на інтервалі
def find_min(f, a, b, eps = 1e-4, factor = 1):
    return bisection_method(f, a, b, eps * factor)

def grad_num_positive(f, x, delta):
    [x1, x2] = x
    [d1, d2] = delta
    f0 = f([x1, x2])
    f1 = f([x1 + d1, x2])
    f2 = f([x1, x2 + d2])
    partx1 = (f1 - f0) / d1
    partx2 = (f2 - f0) / d2
    return [partx1, partx2]

def grad_num_central(f, x, delta):
    [x1, x2] = x
    [d1, d2] = delta
    f0 = f([x1, x2])
    f1p = f([x1 + d1, x2])
    f1m = f([x1 - d1, x2])
    f2p = f([x1, x2 + d2])
    f2m = f([x1, x2 - d2])
    partx1 = (f1p - f1m) / (2 * d1)
    partx2 = (f2p - f2m) / (2 * d2)
    return [partx1, partx2]

def grad_num_negative(f, x, delta):
    [x1, x2] = x
    [d1, d2] = delta
    f0 = f([x1, x2])
    f1 = f([x1 - d1, x2])
    f2 = f([x1, x2 - d2])
    partx1 = (f0 - f1) / d1
    partx2 = (f0 - f2) / d2
    return [partx1, partx2]

def hessian_num_positive(f, x, delta):
    [x1, x2] = x
    [d1, d2] = delta
    [dfx0, dfy0] = grad_num_positive(f, x, delta)
    [dfxpx, dfypx] = grad_num_positive(f, [x1 + d1, x2], delta)
    [dfxpy, dfypy] = grad_num_positive(f, [x1, x2 + d2], delta)
    h11 = (dfxpx - dfx0) / d1
    h12 = (dfypx - dfy0) / d1
    h21 = (dfxpy - dfx0) / d2
    h22 = (dfypy - dfy0) / d2
    return [[h11, h12], [h21, h22]]

def hessian_num_central(f, x, delta):
    [x1, x2] = x
    [d1, d2] = delta
    f0 = f(x)
    fpp = f([x1 + d1 / 2, x2 + d2 / 2])
    fpm = f([x1 + d1 / 2, x2 - d2 / 2])
    fmp = f([x1 - d1 / 2, x2 + d2 / 2])
    fmm = f([x1 - d1 / 2, x2 - d2 / 2])
    f1p = f([x1 + d1, x2])
    f1m = f([x1 - d1, x2])
    f2p = f([x1, x2 + d2])
    f2m = f([x1, x2 - d2])
    h11 = (f1p - 2 * f0 + f1m) / d1 ** 2
    h12 = (fpp - fpm - fmp + fmm) / d1 / d2
    h21 = h12
    h22 = (f2p - 2 * f0 + f2m) / d2 ** 2
    return [[h11, h12], [h21, h22]]
    
def hessian_num_negative(f, x, delta):
    [x1, x2] = x
    [d1, d2] = delta
    [dfx0, dfy0] = grad_num_negative(f, x, delta)
    [dfxmx, dfymx] = grad_num_negative(f, [x1 - d1, x2], delta)
    [dfxmy, dfymy] = grad_num_negative(f, [x1, x2 - d2], delta)
    h11 = (dfx0 - dfxmx) / d1
    h12 = (dfx0 - dfymx) / d1
    h21 = (dfx0 - dfxmy) / d2
    h22 = (dfx0 - dfymy) / d2
    return [[h11, h12], [h21, h22]]


# First derivative
def D1(f, h):
    return lambda x: (f(x+h) - f(x-h)) / h # Simplest symmetric

# Second derivative
def D2(f, h):
    return lambda x: (f(x+h) - 2 * f(x) + f(x-h)) / h**2 # Simplest symmetric

def newton_method(f, x0, eps = 1e-30, verbose = False, exit = "norm", iterations = 1000, analytic_derivative = True, gradf = (lambda x: grad_s6_f2(x)), hessian_f = (lambda x: hessian_s6_f2(x)), derivative_scheme = "central", h = [1e-6, 1e-6], optimize_lambda = True):
    xs = []
    xs.append(x0)
    i = 0
    def exit_criteria_check():
        shouldExit = False
        if(exit == "norm"):
            shouldExit = shouldExit or (norm(grad_s6_f2(xs[-1])) < eps)
        if(exit == "delta"):
            shouldExit = shouldExit or (((norm(vvsubtract(xs[-1], xs[-2])) / norm(xs[-2])) < eps) and ((f(xs[-2]) - f(xs[-1])) < eps))

        shouldExit = shouldExit or (i >= iterations) # Безумовне закінчення після iterations ітерацій
        return shouldExit
        
    
    while True:
        if(analytic_derivative):
            gf = gradf(x0)
            H = hessian_f(x0)
        elif derivative_scheme == "central":
            gf = grad_num_central(f, x0, h)
            H = hessian_num_central(f, x0, h)
        elif derivative_scheme == "positive":
            gf = grad_num_positive(f, x0, h)
            H = hessian_num_positive(f, x0, h)
        elif derivative_scheme == "negative":
            gf = grad_num_negative(f, x0, h)
            H = hessian_num_negative(f, x0, h)
            
        invH = Inverse22(H)
        if verbose:
            print(f'H^(-1): {invH}')
        if not IsPositiveM(invH):
            [[h11, h12], [h21, h22]] = H
            if(h11 < 0 or h22 < 0):
                print(f'\tAt {i}: Not positive diagonal in inverse Hessian. Divergence')
                break;
            # C = [[math.sqrt(abs(h11)), 0], [0, math.sqrt(abs(h22))]]
            invC = Inverse22([[math.sqrt(abs(h11)), 0], [0, math.sqrt(abs(h22))]])
            # invP = mmmult(C, mmmult(invH, C))
            P = mmmult(invC, mmmult(H, invC))
            [[a, b], [c, d]] = P
            alpha1 = (a + d + math.sqrt((a + d)**2 + 4*b*c)) / 2.0
            alpha2 = (a + d - math.sqrt((a + d)**2 + 4*b*c)) / 2.0
            e1 = ([0, 1] if d == alpha1 else [1, -c/(d-alpha1)])
            e2 = ([0, 1] if d == alpha2 else [1, -c/(d-alpha2)])
            e1 = constv(1.0 / norm(e1), e1)
            e2 = constv(1.0 / norm(e2), e2)
            alpha1 = max(abs(alpha1), 1e-100)
            alpha2 = max(abs(alpha2), 1e-100)
            invPTilde = mmadd(constm(1.0 / alpha1, vvouter(e1, e1)), constm(1.0 / alpha2, vvouter(e2, e2)))
            print(f'\tAt {i}: Зведення оберненої матриці Гессе до датньо визначеного виду')
            # Зведена до датньо визначеного виду обернена матриця Гессе
            invH = mmmult(mmmult(invC, invPTilde), invC)
            
        s = constv( -1.0, mvmult(invH, gf))
        if norm(s) == 0.0:
            print(f'\tBreak at {i}: direction vector is zero')
            break
        if verbose:
            print(f's = {s}')
        if optimize_lambda:
            shat = constv(1.0 / norm(s), s)
            if verbose:
                print(f'shat = {shat}')
            fl = lambda l: f(vvadd(x0, constv(l, shat)))
            ll, lr = find_interval(fl, 0)
            if verbose:
                print(f'lambda*_left = {ll}, lambda*_right = {lr}')
            l = find_min(fl, ll, lr)
            if verbose:
                print(f'lambda*_opt = {l}')
            if verbose:
                print(f'lambda_opt = {l / norm(s)}')
            # l0 = newton_method_for_zero(D1(fl, 1e-6), 0, 1)
            x0 = vvadd(x0, constv(l, shat))
        else:
            # print(f'i = {i}')
            x0 = vvadd(x0, s)
        if verbose:
            print(f'x{i+1} = {x0}')
        if verbose:
            print(f"f(x{i+1}) = {s6_f2(x0)}")

        i = i + 1
        xs.append(x0)
        if f(xs[-2]) - f(xs[-1]) < 0.0:
            print(f'\tBreak at {i}: divergence')
            break
        if exit_criteria_check():
            break
        

    if verbose:
        print(f'\nПошук закінчено:')
        print(f'\tx = {x0}')
        print(f'\tf(x) = {s6_f2(x0)}')
        print(f'\t|gradf(x)| = {norm(grad_s6_f2(x0))}')
        print(f'\tH^(-1)(x) = {Inverse22(hessian_s6_f2(x0))}')
    return i, xs

def s6_f2(x):
    [x1, x2] = x
    return (10 * (x1 -x2)**2 + (x1 - 1)**2)**4

def grad_s6_f2(x):
    [x1, x2] = x
    partx1 = 4 * (10 * (x1 -x2)**2 + (x1 - 1)**2)**3 * (20 * (x1 -x2) + 2 * (x1 - 1))
    partx2 = 4 * (10 * (x1 -x2)**2 + (x1 - 1)**2)**3 * (-20 * (x1 -x2))
    return [partx1, partx2]

def hessian_s6_f2(x):
    [x1, x2] = x
    # partx1 = 4 * (10 * (x1 -x2)**2 + (x1 - 1)**2)**3 * (20 * (x1 -x2) + 2 * (x1 - 1))
    h11 = 12 * (10 * (x1 -x2)**2 + (x1 - 1)**2)**2 * (20 * (x1 -x2) + 2 * (x1 - 1))**2 + 4 * (10 * (x1 -x2)**2 + (x1 - 1)**2)**3 * 22
    h12 = 12 * (10 * (x1 -x2)**2 + (x1 - 1)**2)**2 * (20 * (x1 -x2) + 2 * (x1 - 1)) * (-20 * (x1 -x2)) + 4 * (10 * (x1 -x2)**2 + (x1 - 1)**2)**3 * (-20)
    h21 = h12
    h22 = 12 * (10 * (x1 -x2)**2 + (x1 - 1)**2)**2 * (-20 * (x1 -x2))**2 + 4 * (10 * (x1 -x2)**2 + (x1 - 1)**2)**3 * (20)
    return [[h11, h12], [h21, h22]]

def s6_g1(x):
    [x1, x2] = x
    return max(1 - (x1 + 1)**2 - x2**2, 1e-100)

def s6_grad_gm1(x):
    [x1, x2] = x
    partx1 = 2 * (x1 + 1) / g(x)**2
    partx2 = 2 * x2 / g(x)**2
    return [partx1, partx2]

# Гессіан для 1/g, для коли функція g одна з s6_g1 або s6_g3 
def s6_hessian_gm1(x):
    [x1, x2] = x
    h11 = 2 / g(x)**2 + 8 * (x1 + 1)**2 / g(x)**3
    h12 = 8 * (x1 + 1) * x2 / g(x)**3
    h21 = 8 * (x1 + 1) * x2 / g(x)**3
    h22 = 2 / g(x)**2 + 8 * (x2)**2 / g(x)**3
    return [[h11, h12], [h21, h22]]

def s6_g2(x):
    [x1, x2] = x
    return max(9 - (x1 + 1)**2 - x2**2, 1e-100)

def s6_g3(x):
    [x1, x2] = x
    return max(4 - x1 - x2, 1e-100)

def s6_grad_g3m1(x):
    [x1, x2] = x
    partx1 = 1 / g(x)**2
    partx2 = 1 / g(x)**2
    return [partx1, partx2]

def s6_hessian_g3m1(x):
    [x1, x2] = x
    h11 = 2 / g(x)**3
    h12 = 2 / g(x)**3
    h21 = 2 / g(x)**3
    h22 = 2 / g(x)**3
    return [[h11, h12], [h21, h22]]

def plotFunc(f,xs):
    # Sample data
    x_values = [x[0] for x in xs]
    y_values = [x[1] for x in xs]

    plt.figure(figsize=(6, 6))
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', label='Line Plot')

    # contour_levels = np.linspace(-1, 1, 10)
    # plt.contour(X, Y, Z, levels=contour_levels, cmap='viridis')

    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Траєкторія пошуку($h=0.1$, $\\varepsilon=10^{-30}$)')
    plt.grid(True)
    plt.legend()
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1, 2)

    plt.show()

if '__main__' == __name__:
    x0, x1 = [-1.2, 0.0], [1, 1]
    print(f's6_f2({x0}) = {s6_f2(x0)}')
    print(f's6_f2({x1}) = {s6_f2(x1)}')
    # newton_method(s6_f2, x0)
    print(f'gradf({x0}) = {grad_s6_f2(x0)}')
    print(f'gradf_num({x0}) = {grad_num_central(s6_f2, x0, [1e-6, 1e-6])}')
    print(f'hessianf({x0}) = {hessian_s6_f2(x0)}')
    print(f'hessianf_num({x0}) = {hessian_num_central(s6_f2, x0, [1e-6, 1e-6])}')

    print(f'\nМетод Ньютона для eps = 1e-30:')
    it, xs = newton_method(s6_f2, [-1.2, 0.0], eps = 1e-30, verbose = True)
    x = xs[-1]

    print(f'\nx{it} = {x}')
    print(f'f(x{it}) = {s6_f2(x)}')
    print(f'|gradf(x{it})| = {norm(grad_s6_f2(x))}')
    print(f'H^(-1)(x{it}) = {Inverse22(hessian_s6_f2(x))}')
    
    print(f'\nДослідження збіжності в залежності від довжини кроку при обчисленні похідних')
    eps = 1e-30
    print(f'eps: {eps}')
    print(f'Критерій закінчення: |grad f| < eps')
    for i in range(25):
        h = 10**(5 -i)
        it, xs = newton_method(s6_f2, [-1.2, 0.0], eps = eps, analytic_derivative = False, derivative_scheme = "central", h = [h, h])
        print(f'h = {h}: steps = {it}')
        if i==6:
            plotFunc(s6_f2, xs)

    print(f'\nДослідження збіжності в залежності від схеми обчислення похідних')
    eps = 1e-20
    print(f'eps: {eps}')
    print(f'Критерій закінчення: |grad f| < eps')
    for scheme in ["positive", "central", "negative"]:
        print(f'Схема: {scheme}')
        for i in range(10):
            h = 10**(-i)
            it, xs = newton_method(s6_f2, [-1.2, 0.0], eps = eps, analytic_derivative = False, derivative_scheme = scheme, h = [h, h])
            print(f'h = {h}: steps = {it}')

    print(f'\nДослідження збіжності в залежності від критерія закінчення')
    eps = 1e-30
    h = 1e-4
    scheme = "central"
    print(f'eps: {eps}')
    print(f'h: {h}')
    print(f'Схема: {scheme}')
    # print(f'Крітерій закінчення: |grad f| < eps')
    for exit_criteria in ["norm", "delta"]:
        print(f'Критерій закінчення: {exit_criteria}')
        it, xs = newton_method(s6_f2, [-1.2, 0.0], eps = eps, exit = exit_criteria, analytic_derivative = False, derivative_scheme = scheme, h = [h, h])
        x = xs[-1]
        print(f'h = {h}: steps = {it}')
        print(f'\tx{it} = {x}')
        print(f'\tf(x{it}) = {s6_f2(x)}')
        print(f'\t|gradf(x{it})| = {norm(grad_s6_f2(x))}')
        print(f'\tH^(-1)(x{it}) = {Inverse22(hessian_s6_f2(x))}')

    print(f'\nМетод штрафних функцій')
    print(f'1. Мінімум поза допустимою області')
    g = s6_g1
    x = [-1.2, 0.0]
    for i in range(15):
        r = 10**(-i)
        P = lambda x: s6_f2(x) + r / g(x)
        gradP = lambda x: vvadd(grad_s6_f2(x), constv(r, s6_grad_gm1(x)))
        hessianP = lambda x: mmadd(hessian_s6_f2(x), constm(r, s6_hessian_gm1(x)))
        it, xs = newton_method(P, x, eps = 1e-20, exit="delta", gradf=gradP, hessian_f=hessianP)
        x = xs[-1]
        print(f'r = {r}: steps = {it}')
        print(f'\tx{it} = {x}')
        print(f'\tg(x{it}) = {g(x)}')
        print(f'\t|gradP(x{it})| = {norm(gradP(x))}')
        print(f'\tgradf(x{it}) = {norm(grad_s6_f2(x))}')
        print(f'\tr grad(g^-1)(x{it}) = {r * norm(s6_grad_gm1(x))}')
    print(f'\tf(x{it}) = {s6_f2(x)}')
    print(f'\tg(x{it}) = {g(x)}')
    print(f'1. Мінімум всередині допустимої області')
    g = s6_g2
    x = [-1.2, 0.0]
    for i in range(15):
        r = 10**(-i)
        P = lambda x: s6_f2(x) + r / g(x)
        gradP = lambda x: vvadd(grad_s6_f2(x), constv(r, s6_grad_gm1(x)))
        hessianP = lambda x: mmadd(hessian_s6_f2(x), constm(r, s6_hessian_gm1(x)))
        it, xs = newton_method(P, x, eps = 1e-20, exit="delta", gradf=gradP, hessian_f=hessianP)
        x = xs[-1]
        print(f'r = {r}: steps = {it}')
        print(f'\tx{it} = {x}')
        print(f'\tg(x{it}) = {g(x)}')
        print(f'\t|gradP(x{it})| = {norm(gradP(x))}')
        print(f'\tgradf(x{it}) = {norm(grad_s6_f2(x))}')
        print(f'\tr grad(g^-1)(x{it}) = {r * norm(s6_grad_gm1(x))}')
    print(f'\tf(x{it}) = {s6_f2(x)}')
    print(f'\tg(x{it}) = {g(x)}')
    print(f'1. Неопукла область')
    g = s6_g3
    x = [-1.2, 0.0]
    for i in range(15):
        r = 10**(-i)
        P = lambda x: s6_f2(x) + r / g(x)
        gradP = lambda x: vvadd(grad_s6_f2(x), constv(r, s6_grad_g3m1(x)))
        hessianP = lambda x: mmadd(hessian_s6_f2(x), constm(r, s6_hessian_g3m1(x)))
        it, xs = newton_method(P, x, eps = 1e-20, exit="delta", gradf=gradP, hessian_f=hessianP)
        x = xs[-1]
        print(f'r = {r}: steps = {it}')
        print(f'\tx{it} = {x}')
        print(f'\tg(x{it}) = {g(x)}')
        print(f'\t|gradP(x{it})| = {norm(gradP(x))}')
        print(f'\tgradf(x{it}) = {norm(grad_s6_f2(x))}')
        print(f'\tr grad(g^-1)(x{it}) = {r * norm(s6_grad_gm1(x))}')
    print(f'\tf(x{it}) = {s6_f2(x)}')
    print(f'\tg(x{it}) = {g(x)}')