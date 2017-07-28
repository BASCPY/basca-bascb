"""
============================================================
Authors: Helber Giovanny Sissa Becerra
         Tatiana Andrea Higuera Munevar
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The BASC algorithms are binarization techniques that aim
at determining a robust binarization by analyzing the data
at multiple scales
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Method: BASCB
============================================================
"""

import numpy as np
from numpy import *
import math
import random
import types
from scipy.special import iv


def mean(values, a, b):
    """
    This function represents the value of the mean, the mean is
    equal to the sum over every possible value weighted by the
    probability of that value.

    Parameters
    ----------
    :param values: Array
                   Vector that contain set of data
    :param a: Integer
              Number representing where start to calculate the mean
    :param b: Integer
              Number representing where finish to calculate the mean

    Returns
    -------
    :return mean_val: Integer
                      The mean in the vector calculated from a
                      through b
    """
    begin = a - 1
    end = b - 1
    mean_val = 0.0
    for i in range(begin, end + 1):
        mean_val += values[i]
    mean_val /= (end - begin + 1.0)
    return mean_val


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
    fhalf, shalf = num[low:mid], num[mid:hi]
    i, j = 0, 0
    for k in range(low, hi):
        if i == mid - low:
            num[k] = shalf[j]
            j += 1
        elif j == hi - mid:
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


class BASCB:
    def __init__(self, u):
        self.u = u[:]
        self.uSort = sortedFunction(u[:])
        self.deriv, self.deriTotal = self.calculateSlopes()
        self.sigma = [round(i * 0.1, 3) for i in range(1, 201)]
        self.smoothed = [[0 for j in range(int(len(self.u)) - 1)] for i in range(len(self.sigma))]
        self.zerocrossing = [[0 for j in range(int(len(self.u) / 2))] for i in range(len(self.sigma))]
        self.funtionSteps(self.sigma)

        self.steps = []
        self.index = []
        self.greatest_index_ind = 0
        self.greatest_steps_col = 0
        self.greatest_steps_row = 0
        self.getQuantizatizations()

        self.meanlist = [[0 for j in range(int(len(self.uSort)))] for i in range(len(self.index) + 1)]
        self.smoothedX = [[0 for j in range(int(len(self.smoothed[0]) + 1))] for i in range(len(self.index))]
        self.h = [[0 for j in range(self.greatest_steps_col)] for i in range(self.greatest_steps_row)]
        self.v = [0 for j in range(len(self.index))]
        self.calcDiscontinuity()

        self.uBinario, self.t, self.p = self.locationVariation()
        self.hTotal = list()
        self.xPuntos = self.graph()

    def memcmp(self, temp, temp1, length):
        """
        This function makes a selection in the vector zerocrossing
        points to obtain the steps, comparing that the zerocrossing
        temp and temp1 that have not been added later to the steps

        Parameters
        ----------
        :param temp: Array
                     Vector that contain the zerocrossing
        :param temp1: Array
                     Vector that contain the zerocrossing
        :param length: Integer
                     Number representing the length of the array
                     to be compared

        Returns
        -------
        :return flag : Boolean
                     This boolean representing whether to add
                     zerocrossing to steps
        """
        flag = False
        if temp[length] != 0 and temp1[length] == 0:
            flag = True
        return flag

    def calculateSlopes(self):
        """
        This function calculates the scale space

        :return derif : Array
                        The slopes this is the first derivatives of
                        the original step function
        :return derifTotal : Integer
                             This value is the sum of the values in
                             derif
        """
        derifTotal = 0
        derif = [0 for i in range(len(self.uSort) - 1)]
        for i in range(len(self.uSort) - 1):
            derif[i] = self.uSort[i + 1] - self.uSort[i]
            derifTotal += derif[i]

        return derif, derifTotal

    def funtionSteps(self, sigma):
        """
        This function compute  for each function the smoothed slope,
        where each maximum x in the set is the location of a discontinuity.
        Finding a function with a single remaining discontinuity.

        :param sigma: Array
                      number of smoothing parameters
        """
        for i in range(len(sigma)):
            sTimesTwo = 2.0 * sigma[i]
            e_pow_mstt = math.exp(-sTimesTwo)
            for k in range(len(self.deriv)):
                sum = 0.0
                for j in range(len(self.deriv)):
                    bessel = iv(float(k - j), sTimesTwo)
                    sum += self.deriv[j] * e_pow_mstt * bessel
                self.smoothed[i][k] = sum / self.deriTotal

            k = 0
            for j in range(len(self.smoothed[0])):
                BanZerocrossing = 0
                if (j == 0) and (self.smoothed[i][j] > self.smoothed[i][j + 1]):
                    BanZerocrossing += 1
                elif (j == len(self.smoothed[0]) - 1) and (self.smoothed[i][j - 1] < self.smoothed[i][j]):
                    BanZerocrossing += 1
                elif (j > 0) and (j < len(self.smoothed[0]) - 1) and (
                    self.smoothed[i][j - 1] < self.smoothed[i][j]) and (self.smoothed[i][j] > self.smoothed[i][j + 1]):
                    BanZerocrossing += 1
                if BanZerocrossing == 1:
                    self.zerocrossing[i][k] = j + 1
                    k += 1

            if k == 0:
                for j in range(len(self.zerocrossing[0])):
                    self.zerocrossing[i][j] = j + 1

    def getQuantizatizations(self):
        """
        This function separates the steps of zerocrossing, obtaining the
        possible candidates for the step function, where each zerocrossing
        maximum in the set is the location of a discontinuity, thus obtaining
        discontinuities
        """

        current = 0
        qrPos = 0
        length = len(self.zerocrossing[0])-1
        for cross in range(len(self.zerocrossing)-1):
            if cross == 0 or self.memcmp(self.zerocrossing[cross][:], self.zerocrossing[cross+1][:],length-1):
                self.steps.insert(0, self.zerocrossing[cross])
                self.index.append(self.zerocrossing.index(self.zerocrossing[cross])+1)
                qrPos += 1
                current = cross
                length -= 1
                if not (self.zerocrossing[current][1]):
                    cross = len(self.zerocrossing[0][:])
            elif length-1 == 0:
                self.steps.insert(0, self.zerocrossing[cross])
                self.index.append(cross + 1)
                qrPos += 1
                current = cross
                length -= 1
                if not (self.zerocrossing[current][1]):
                    cross = len(self.zerocrossing[0][:])
                break

        self.greatest_steps_row = qrPos
        self.greatest_index_ind = qrPos

        self.index = sorted(self.index, reverse=True)

        for i in range(len(self.zerocrossing[0])):
            if (i == len(self.zerocrossing[0]) or (self.zerocrossing[0][i] == 0)):
                self.greatest_steps_col = i
                i = len(self.zerocrossing[0]) + 1

    def calcDiscontinuity(self):
        """
        This function compute a series of step functions
        step functions is obtained by rearranging the
        original time series measurements in increasing order.
        Then, step functions with fewer discontinuities
        are calculated.
        """

        smoothed = [0 for j in range(int(len(self.smoothed[0]) + 1))]
        step_heights = [0 for j in range(int(len(self.meanlist[0]) - 1))]
        self.meanlist[len(self.meanlist) - 1] = self.uSort

        for i in range(len(self.v)):
            smoothedSlopes = self.smoothed[self.index[i] - 1]
            smoothed[0] = self.uSort[0]
            for j in range(1, len(smoothed)):
                smoothed[j] = smoothed[j - 1] + smoothedSlopes[j - 1]

            self.smoothedX[i] = smoothed[:]

            for j in range(len(self.steps[0])):
                if self.steps[i][j] != 0:
                    if j == 0:
                        self.meanlist[i][j] = mean(smoothed, 1, self.steps[i][j])
                    else:
                        self.meanlist[i][j] = mean(smoothed, self.steps[i][j - 1] + 1, self.steps[i][j])
                        step_heights[j - 1] = self.meanlist[i][j] - self.meanlist[i][j - 1]
                        self.h[i][j - 1] = step_heights[j - 1]
                else:
                    break

            if j <= len(self.steps[0]):
                self.meanlist[i][j] = mean(smoothed, self.steps[i][j - 1] + 1, len(smoothed))
                step_heights[j - 1] = self.meanlist[i][j] - self.meanlist[i][j - 1]
                self.h[i][j - 1] = step_heights[j - 1]
            max_quot = -1.0
            max_quot_ind = -1
            for j in range(len(self.steps[0])):
                if self.steps[i][j] != 0:
                    idx = self.steps[i][j]
                    mn = (smoothed[idx] + smoothed[idx - 1]) * 0.5
                    cur_quot = step_heights[j] / self.cost(smoothed, 0, len(smoothed) - 1, mn)
                    if cur_quot > max_quot:
                        max_quot = cur_quot
                        max_quot_ind = j

                else:
                    break
            self.v[i] = self.steps[i][max_quot_ind]

    def cost(self, vect, a, b, y):
        """
        This function determine the cost(a, b) of a function,
        adding the costs of all the steps in the function

        Parameter
        ---------
        :param vect : Array
                      List of point
        :param a : Integer
                   Start point
        :param b : Integer
                   End point
        :param y : Integer
                   Point

        Returns
        -------
        :return r : Float
                    The quadratic distance of the values of f
                    between a and b
        """

        cost = 0
        for i in range(a, b + 1):
            cost_root = vect[i] - y
            cost += cost_root * cost_root
        return cost

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

        floorM = int(math.floor(median(self.v)))
        t = (self.uSort[floorM] + self.uSort[floorM - 1]) / 2
        uBinario = []
        strBinario =""
        for i in self.u:
            if i <= t:
                uBinario.append(0)
                strBinario += "0"
            else:
                strBinario += "1"
                uBinario.append(1)

        self.str_binario = strBinario
        samples = [0 for j in range(len(self.u))]
        nom = self.normabsmedian(self.u)
        tau = 0.01
        numberOfSamples = 999
        t_zero = tau - nom
        p = 1.0

        for i in range(numberOfSamples):
            self.blockwiseboot(samples)
            mdm = self.normabsmedian(samples)
            t_star = nom - mdm
            if t_star >= t_zero:
                p += 1.0

        p /= float(numberOfSamples + 1)
        return uBinario, t, p

    def normabsmedian(self, samples):
        """
        This function computes the average deviation for an array
        from its median

        Parameter
        ---------
        :param samples: Array
                        Lista of median

        Returns
        -------
        :return result: Integer
                        The average deviation
        """
        mem = [0 for j in range(len(self.u))]
        median_val = median(samples)
        for i in range(len(mem)):
            mem[i] = math.fabs(median_val - float(samples[i]))

        mean_val = mean(mem, 1, len(mem))

        result = mean_val / (len(self.uSort) - 1)

        return result

    def blockwiseboot(self, samples):
        """
        This function resamples the original values. It takes
        ceil(#elements / bl) blocks of length bl = round((#elements)^(0.25))+1
        out of the original * values and concatenates them to a new vector
        """

        temp = math.sqrt(math.sqrt(float(len(samples))))
        bl = math.floor(temp) + 1

        sample_count = math.ceil(float(len(samples)) / float(bl))

        index = 0
        max = float(len(samples) - bl)
        values = []
        for i in range(sample_count):
            rando = random.uniform(-0.5, max + 0.5)

            rando = math.floor(rando)

            for j in range(bl):
                if index < len(samples):
                    values.append(samples[int(rando + j)])
                index += 1

    def graph(self):
        """
        This function organize the break point to plot
        """

        i = 0
        xPuntos = list()
        xPuntos.append([j for j in range(len(self.u) + 1)])
        stepsxAct = xPuntos[-1]
        while i < len(self.zerocrossing):
            stepsxNext = self.zerocrossing[i][:]
            if stepsxAct != stepsxNext:
                tmp = stepsxNext
                xPuntos.append(stepsxNext)
                # xPuntos[-1].append(len(self.u))
            elif stepsxAct[1] == 0:
                i = len(self.zerocrossing)
            i += 1
            stepsxAct = stepsxNext

        for i in range(1, len(xPuntos)):
            j = len(xPuntos[i]) - 1
            while xPuntos[i][j] == 0:
                if xPuntos[i][j] == 0:
                    del xPuntos[i][-1]
                j -= 1
            xPuntos[i].insert(0, 0)
            xPuntos[i].append(len(self.u))

        for i in self.h:
            self.hTotal.append(sum(i))

        for i in range(len(self.u) - 1):
            pass

        return xPuntos