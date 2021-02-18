import numpy as np
import matplotlib.pyplot as plt

def findLocalArgExt(apoly, a, b):
    """
    Function to find the indices at which local minima and maxima occur in the error function 

    apoly: Array of coefficients of approximating polynomial
    a,b: Endpoints of approximation interval [a,b]
    """
    x = np.linspace(a,b,num=1000)
    E_f = np.polyval(np.flip(apoly), x) - (np.exp(x) + x + np.ones(len(x)))
    argmax = []#List of arguments at which the local maxima occur
    argmin = []#List of arguments at which local minima occur
    #Find indices at which local maxima and minima occur
    for i in np.arange(len(x)):
        if i != 0 and i != (len(x) - 1):
            if E_f[i] > E_f[i-1] and E_f[i] > E_f[i+1]:
                argmax.append(x[i])
            elif E_f[i] < E_f[i-1] and E_f[i] < E_f[i+1]:
                argmin.append(x[i])
    if E_f[0] > E_f[1]:
        argmax.append(x[0])
    elif E_f[0] < E_f[1]:
        argmin.append(x[0])

    if E_f[len(x) - 1] > E_f[len(x) - 2]:
        argmax.append(x[len(x) - 1])
    elif E_f[len(x) - 1] < E_f[len(x) - 2]:
        argmin.append(x[len(x) - 1])
        
    result = np.concatenate((np.array(argmax), np.array(argmin)))#Cast to numpy array
    return np.sort(result)

def ensureAlternationProperty(E_arr):
    """
    Function that returns an integer array v indicating which indices to extract from new_test_points
    in the function exchange() to ensure the alternation property holds for the selected test points
    that will be used in the next iteration, i.e. new_test_points[v[i]] will be extracted for i = 0,1,...,len(v) - 1

    E_arr: Values of error function evaluated at test points
    """
    J = len(E_arr)
    j = 0
    v = np.zeros(J, dtype=int)
    xe = E_arr[0]
    xv = 1
    for k in np.arange(1,J):
        if np.sign(E_arr[k]) == np.sign(xe):
            if np.abs(E_arr[k]) > np.abs(xe):
                xe = E_arr[k]#Element of max. value in E_arr
                xv = k
        else:
            v[j] = xv
            j = j + 1
            xe = E_arr[k]
            xv = k
    v[j] = xv
    return v
        
def exchange(sol_vec, prev_test_points, a, b):
    """
    Function to update find the alternation points of the polynomial with coefficients specified by coefs

    sol_vec: Result of solving remez system. Coefficients of approximating polynomial occupy the first N + 1 indices,
             in ascending degree order, in indices 0,1,...,len(coefs) - 2, uniform approximation error term delta
             occupies cell in last index
    prev_test_points: Array of test points before the exchange
    a,b: Endpoints of approximation interval [a,b]

    Returns new array of test points
    """
    N_2 = len(sol_vec)
    approximating_poly = sol_vec[0:(N_2 - 1)]#Coefficients of polynomial
    delta = sol_vec[N_2 - 1]
    new_test_points = findLocalArgExt(approximating_poly, a, b)
    E_arr = np.polyval(np.flip(approximating_poly), new_test_points) - (np.exp(new_test_points) + new_test_points + np.ones(len(new_test_points)))
    v2 = ensureAlternationProperty(E_arr)#Ensure alternation property
    new_test_points = new_test_points[v2]
    #Make sure length of new test points array is correct
    if len(new_test_points) > N_2:
        new_test_points = new_test_points[0:N_2]
    return np.sort(new_test_points)

def solveRemezSystem(test_points, f):
    """
    Function to find the Chebyschev solution of the linear system of euqations that arises
    in the Remez exchange algorithm

    test_points: Test points
    f: Values of function evaluated at test points, we are solving Ax = f

    Returns a vector of length N + 2, the first N + 1 indices are the coefficients of the polynomial
    in ascending degree order, last index is the max. uniform approximation error delta
    """
    size = len(f)
    A = np.zeros((size,size))
    for i in np.arange(size):
        for j in np.arange(size-1):
            A[i][j] = np.power(test_points[i],j)
    for j in np.arange(size):
        A[j][size-1] = np.power(-1.0,j)
    chebyschev_sol = np.linalg.solve(A,f)
    return chebyschev_sol

def remez(N, a, b, tol=1e-3):
    """
    Function to find the minimax interpolating polynomial of degree N to 1/sqrt(1 + x^2) on [a,b]

    N: Degree of interpolating polynomial
    a,b: Endpoints of approximation interval
    tol: Error tolerance in infinity norm, default is 10^-3
    """
    test_points = np.linspace(a, b, num=(N+2))
    f = np.exp(test_points) + test_points + np.ones(len(test_points))#Function being approximated
    E_0 = np.max(np.abs(f))
    E = 1e15
    delta = 0.1
    while (E - delta)/E_0 >= tol:
        #Solve remez system and update delta
        sol_vec = solveRemezSystem(test_points, f)
        approximating_poly = np.flip(sol_vec[0:(len(sol_vec) - 1)])#Polynomial coefficients in ascending degree order
        delta_prev = delta
        delta = sol_vec[len(sol_vec) - 1]#Uniform approximation error
        #Exchange method to find new test points
        test_points = exchange(sol_vec, np.copy(test_points), a, b)
        f = np.exp(test_points) + test_points + np.ones(len(test_points))
        #Update E
        E_arr = np.polyval(approximating_poly, test_points) - f#Aray of values of error function at  test points
        E = np.max(np.abs(E_arr))
        if np.max(np.abs(delta_prev - delta)) < tol:
            break
    poly_coefs = sol_vec[0:(len(sol_vec) - 1)]#Extract coefficients from solution vector
    return poly_coefs
    
def main():
    a = 0
    b = 2.0
    N = 5
    poly = remez(N, a, b, tol=1e-6)
    print(poly)
    x = np.linspace(a,b, num=500)
    f = np.exp(x) + x + np.ones(len(x))
    plt.plot(x, f, color='blue')
    plt.plot(x, np.polyval(np.flip(poly), x), color='green')
    plt.show()

if __name__ == "__main__":
    main()