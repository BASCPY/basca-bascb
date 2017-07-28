"""
============================================================
Authors: Helber Giovanny Sissa Becerra
         Tatiana Andrea Higuera Munevar
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The BASC algorithms are binarization techniques that aim
at determining a robust binarization by analyzing the data
at multiple scales
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Method: BASCA
============================================================
"""

import numpy as np
from numpy import *

import math


def fA(i, u, d):
    """
    This function computes the intervals of function F

    Parameters
    ----------
    :param i : Integer
               This value is the interval in the A function
    :param u: Array
              A vector that elements are sorted in ascending
              order.
    :param d: Array
              Vector discontinuities

    Returns
    -------
    :return r: Array
    """
    r = [None for j in range(2)]
    if i == 1:
        r[0] = 0
        r[1] = d[i]
    elif i == len(u):
        r[0] = d[i - 1]
        r[1] = d[len(u)]
    else:
        r[0] = d[i - 2]
        r[1] = d[i - 1]

    return r


def interval(x, i):
    """
    This function computes the indicator of function A

    Parameters
    ----------
    :param x : Integer
               Value at which to evaluate
    :param i : Integer
               This value is the interval in the A function

    Returns
    -------
    :return r : Integer
                This value is the indicator at x value
    """
    A = fA(i)
    if x in A[1:]:
        r = 1
    else:
        r = 0
    return r


def f(x, u):
    """
    This function defined a discrete, monotonically
    increasing step function f with N steps

    Parameters
    ----------
    :param x : Integer
               Value at which to evaluate
    :param u: Array
              A vector that elements are sorted in ascending
              order.

    Returns
    -------
    :return r : Integer
                This value is the sum of the values in u
                multiplied with the interval function
	"""
    r = 1
    for i in range(1, len(u)):
        r += u[i] * interval(x, i)

    return r


def solve(num, low, hi):
    """
    This function is responsible for performing a merge sort
    is a stable external ordering algorithm based on the divide
    and conquer technique.

    Parameters
    ----------
    :param num: Array
                Vector to order
    :param low: Integer
                Number representing where start to order
    :param hi: Integer
               Number representing where finish to order

    Returns
    -------
    :return num: Array
                 vector in ascending order.
    """
    if low + 1 < hi:
        mid = low + ((hi - low) // 2)
        solve(num, low, mid)
        solve(num, mid, hi)
        num = merge(num, low, mid, hi)
    return num


def merge(num, low, mid, hi):
    """
    This function is responsible for making ordering
    comparisons between two numbers, generating changes
    in the vector.

    Parameters
    ----------
    :param num: Array
                Vector to order
    :param low: Integer
                Number representing where start to order
    :param mid: Integer
                Number representing half of where you are
                ordering
    :param hi: Integer
               Number representing where finish to order

    Returns
    -------
    :return num: Array
                 vector in ascending order.
    """
    fhalf, shalf = num[low:mid], num[mid:hi]  # creamos la copia de nuestra lista en dos partes
    i, j = 0, 0  # iniciamos los contadores
    for k in range(low, hi):
        if i == mid - low:
            num[k] = shalf[j]
            j += 1
        elif (j == hi - mid):
            num[k] = fhalf[i]
            i += 1
        else:
            if fhalf[i] <= shalf[j]:
                num[k] = fhalf[i]
                i += 1
            else:
                num[k] = shalf[j]
                j += 1

    return num


def sortedFunction(u):
    """
    This function order a vector in ascending order.

    Parameters
    ----------
    :param u: vector to be sorted

    Returns
    -------
    :return : vector
    """
    return solve(u, 0, len(u))


def median(V):
    """
    This function represents the value of the variable central
    position in a set of ordered data.

    Parameters
    ----------
    :param V: Array
              Vector that contain set of data

    Returns
    -------
    :return r : Integer
               Value that represent the median
    """
    V = sorted(V)
    x = int(len(V) / 2)
    x -= 1
    if len(V) % 2 == 0:
        r = (V[x] + V[x + 1]) / 2
    else:
        r = V[x + 1]
    return r


class BASCA:
    def __init__(self, u):
        self.u = u[:]
        self.uSort = sortedFunction(self.u[:])
        self.d = [i for i in range(1, len(u))]
        self.C, self.Ind = self.stepFunction()
        self.P = self.breakPoints()
        self.h, self.hTotal = self.h()
        self.q, self.Z = self.highJump()
        self.M, self.I, self.V = self.strongestDiscontinuity()
        self.uBinario, self.t = self.locationVariation()
        self.xPuntos = self.graphic()

    def getP(self):
        """
        This function return the breakpoints
        :return P: Array
                   The break points for optimal F function
        """
        return self.P

    def Y(self, a, b):
        """
        This function calculates the step to the mean of the
        values between the start point and the end point

        Parameters
        ----------
        :param a : Integer
                   Start point
        :param b : Integer
                   End point

        Returns
        -------
        :return y : Integer
                    The mean of the values between a and b
        """
        Fi = 0
        for i in range(b, a - 1, -1):
            Fi += self.uSort[i]
        y = Fi / (b - a + 1)

        return y

    def cost(self, a, b):
        """
        This function determine the cost(a, b) of a function,
        adding the costs of all the steps in the function

        Parameter
        ---------
        :param a : Integer
                   Start point
        :param b : Integer
                   End point

        Returns
        -------
        :return r : Float
                    The quadratic distance of the values of f
                    between a and b
        """
        y = self.Y(a, b)
        r = 0.0
        for i in range(a, b + 1):
            r += (self.uSort[i] - y) ** 2
        return r

    def stepFunction(self):
        """
        This function compute a series of step functions
        step functions is obtained by rearranging the
        original time series measurements in increasing order.
        Then, step functions with fewer discontinuities
        are calculated.

        Returns
        -------
        :return C : Array
                    The cost of a function
        :return Ind : Array
                      The indices of the break points
                      in a matrix Ind.
        """
        C = [[0.000 for i in range(len(self.uSort) - 1)] for j in range(len(self.uSort))]
        Ind = [[-1 for i in range(len(self.uSort) - 1)] for j in range(len(self.uSort) - 1)]
        for i in range(0, len(self.uSort)):
            C[i][0] = self.cost(i, len(self.uSort) - 1)

        for j in range(1, len(self.uSort) - 1):
            for i in range(0, len(self.uSort) - j):
                min = 10 << 123
                k = 0
                for d in range(i, len(self.uSort) - j):
                    cTemp = self.cost(i, d) + C[d + 1][j - 1]
                    if min > cTemp:
                        index = k
                        min = cTemp
                    k += 1
                C[i][j] = min
                Ind[i][j - 1] = index + i
        return C, Ind

    def breakPoints(self):
        """
        This function compute the break points of all
        optimal step function

        Returns
        -------
        :return P : Array
                    The break points for optimal F function
        """
        P = [[-1 for i in range(len(self.uSort) - 2)] for j in range(len(self.uSort) - 2)]
        for j in range(0, len(P)):
            z = j
            P[0][j] = self.Ind[0][z]
            if j > 0:
                z -= 1
                for i in range(1, j + 1):
                    P[i][j] = self.Ind[P[i - 1][j] + 1][z]
                    z -= 1
        return P

    def h(self):
        """
        This function compute the difference in height between
        the start point and the end point of the discontinuity.

        Returns
        -------
        :return h : Array
                   The difference of these two mean values.
        :return hTotal : Array
                        The sum total of the difference of
                        these two mean values.
        """
        h = [[0 for i in range(len(self.uSort) - 2)] for j in range(len(self.uSort) - 2)]

        hTotal = [0 for j in range(len(self.uSort) - 2)]
        for j in range(0, len(h)):
            total = 0
            for i in range(0, j + 1):
                if i == 0 and j == 0:
                    h[i][j] = self.Y(self.P[i][j] + 1, len(self.uSort) - 1) - self.Y(0, self.P[i][j])
                elif i == j and j >= 1:
                    h[i][j] = self.Y(self.P[i][j] + 1, len(self.uSort) - 1) - self.Y(self.P[i - 1][j] + 1, self.P[i][j])
                elif i == 0 and j >= 1:
                    h[i][j] = self.Y(self.P[i][j] + 1, self.P[i + 1][j]) - self.Y(0, self.P[i][j])
                else:
                    h[i][j] = self.Y(self.P[i][j] + 1, self.P[i + 1][j]) - self.Y(self.P[i - 1][j] + 1, self.P[i][j])
                total += h[i][j]
            hTotal[j] = total
        return h, hTotal

    def highJump(self):
        """
        This function compute the two criteria in h and e are
        combined into a scoring function

        Returns
        -------
        :return q : Array
                    q is achieved by a high jump size in
                    combination with a low approximation error.
        :return Z : Array
                    The sum of the quadratic distance of all data points
        """
        Z = [[0 for i in range(len(self.uSort) - 2)] for j in range(len(self.uSort) - 2)]
        q = [[0 for i in range(len(self.uSort) - 2)] for j in range(len(self.uSort) - 2)]
        for j in range(0, len(Z)):
            for i in range(0, j + 1):
                Z[i][j] = (self.uSort[self.P[i][j]] + self.uSort[self.P[i][j] + 1]) / 2
        E = [[1 for i in range(len(self.uSort) - 2)] for j in
             range(len(self.uSort) - 2)]
        for j in range(0, len(E)):
            for i in range(0, j + 1):
                e = 0
                for d in range(len(self.uSort)):
                    e += (self.uSort[d] - Z[i][j]) ** 2
                E[i][j] = e
                q[i][j] = self.h[i][j] / E[i][j]
        return q, Z

    def strongestDiscontinuity(self):
        """
        This function the strongest discontinuities of the
        optimal step functions

        Returns
        -------
        :return M : Array
                   Contains the value of the strongest
                   discontinuities for each amount of steps
        :return I : Array
                   Contains the indexes of the strongest
                   discontinuities for each amount of steps
        :return V : Array
                   Contains the maximum value of the strongest
                   discontinuities for each amount of steps
        """
        M = []
        I = []
        for i in range(len(self.q)):
            max = float('-inf')
            for j in range(len(self.q[0])):
                if self.q[j][i] > max:
                    max = self.q[j][i]
                    index = j
            M.append(max)
            I.append(index)
        V = [0 for j in range(len(self.uSort) - 2)]
        for j in range(0, len(V)):
            V[j] = self.P[I[j]][j]
        return M, I, V

    def locationVariation(self):
        """
        This function compute the estimate location and variation
        of the strongest discontinuities
        Returns
        -------
        :return uBinario : Array
                          Define the single binarization threshold
        :return t : Array
                          Define the single binarization threshold
        """
        floorM = int(math.floor(median(self.V)))
        t = (self.uSort[floorM + 1] + self.uSort[floorM]) / 2
        uBinario = []
        strBinario = ""
        for i in self.u:
            if i <= t:
                uBinario.append(0)
                strBinario += "0"
            else:
                strBinario += "1"
                uBinario.append(1)

        self.str_binario = strBinario

        return uBinario, t

    def graphic(self):
        """
        This function organize the break point to plot
        """

        xPuntos = [[] for j in range(len(self.Ind) - 1)]

        for i in range(len(self.P)):
            k = i
            xPuntos[i].append(0)
            for j in range(i + 1):
                xPuntos[i].append(self.P[j][i] + 1)
            xPuntos[i].append(len(self.uSort))
        xPuntos.append(list(range(len(self.uSort) + 1)))
        return xPuntos
