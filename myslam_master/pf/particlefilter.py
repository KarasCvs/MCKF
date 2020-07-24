# SIR particle filter
import numpy as np
import math
from numpy.random import uniform, normal, randint


class PF():
    def __init__(self):
        self.r_noise = 3

    def read_data(self, dimension):
        self.dimension = dimension      # 提议分布的维度
        # self.N = data.shape[1]      # length of data

    def ss(self, A, b, c, d):
        self.A = A
        self.b = b
        self.c = c
        self.d = d

    def parmet(self, r, n=20):
        self.noise = r
        self.particles_number = n

    def resample(self):
        for i in range(self.particles_number):
            wmax = 2*np.max(self.w)*uniform()
            index = randint(0, self.particles_number)
            while(wmax > self.w[index]):
                wmax = wmax - self.w[index]
                index = index + 1
                if index >= self.particles_number:
                    index = 0
        self.particles[:, i] = self.particles[:, index]

    def calculation(self, proposals):
        self.w = np.zeros(self.particles_number)
        self.particles = np.matrix(np.zeros((self.dimension, self.particles_number)))      # initialize particles
        for i in range(self.particles_number):
            self.particles[:, i] = proposals + normal(0, self.noise, (self.dimension, 1))
            self.w[i] = np.sum(np.multiply((1/self.noise/math.sqrt(2*math.pi)),
                                           np.power(np.square(np.linalg.norm(proposals-self.particles[:, i]))/2/self.noise, math.e)))
        wsum = np.sum(self.w)
        self.w = self.w/wsum
        resampling_trager = 1       # resampling
        if resampling_trager:
            self.resample()
        PCenter = np.sum(self.particles, axis=1) / self.particles_number
        return PCenter
