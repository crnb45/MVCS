# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 12:52:04 2024

@author: Henri
"""

import numpy as np

class CommunicationModel:
    sigma_z = 0.1
    sigma_h = 1
    P = 1
    R = 2
    beta = 2
    d = 1000
    
    #Constructor
    def __init__(self, sigma_z, sigma_h, P, R, beta, d, delta, c):
        self.sigma_z = sigma_z
        self.sigma_h = sigma_h
        self.P = P
        self.R = R
        self.beta = beta
        self.d = d
        self.delta = delta
        self.c = c
    
    #Takes a matrix of model updates as input and
    #returns the arithmetic mean as a vector 
    def perfect(self, update_matrix):
        n = update_matrix.shape[1]
        ret = np.sum(update_matrix, 1)/n
        
        return ret
    
    #Communicates one bit of information for each entry and recreates
    #the answer by majority vote. No compression
    def alphan(self, update_matrix):
        #Quantize
        update_bits = np.sign(update_matrix)
        
        #Vote in one of two orthogonal resources (or cast no vote)
        neg_votes = np.where(update_bits < 0, 1, 0)
        pos_votes = np.where(update_bits > 0, 1, 0)
        
        #Generate path loss
        n = update_bits.shape[1]
        l = self._generate_pathloss(n, self.R, self.beta)
        
        #Select transmit power
        p = (2*self.P*np.min(l)**2/l**2).T
        
        #Communicate bits over channel
        y_neg = self._MAC(neg_votes*np.sqrt(p), l)
        y_pos = self._MAC(pos_votes*np.sqrt(p), l)
        
        #Energy comparison to form bits at receiver side
        rx_bits = np.where(np.abs(y_neg) > np.abs(y_pos), -1, 1)
        
        return rx_bits
    
    def _generateMeasureMatrix(self, T):
        if not hasattr(self, 'M'):
            self.M = np.random.randn(T, self.d);
    
    #Communicates one bit of information for each entry and recreates the
    #answer by majority vote. 1-bit compressed sensing before communication.
    #The dimension after compression is T, but 2T channel uses are required
    def henrik(self, update_matrix, T):
        #Will only be generated if this is the first call
        self._generateMeasureMatrix(T)
        
        #Normalize the vectors before transmitting
        n = update_matrix.shape[1]
        for i in range(n):
            update_vec = update_matrix[:, i]
            update_vec = update_vec/np.linalg.norm(update_vec)
            update_matrix[:, i] = update_vec
            
        #Compress
        S = self.M@update_matrix
        
        #Generate path loss
        l = self._generate_pathloss(n, self.R, self.beta)
        
        #Select transmit power
        p = (np.sqrt(2*np.pi)*self.P*np.min(l)**2/l**2).T
        
        #Vote in one of two orthogonal resources (or cast no vote)
        neg_votes = np.where(S < 0, np.sqrt(np.abs(S)*p), 0)
        pos_votes = np.where(S > 0, np.sqrt(np.abs(S)*p), 0)
        
        #Communicate bits over channel
        y_neg = self._MAC(neg_votes, l)
        y_pos = self._MAC(pos_votes, l)
        
        #Energy comparison to form bits at receiver side
        b = np.where(np.abs(y_neg) > np.abs(y_pos), -1, 1)
        
        #Passive reconstruction
        x_hat = self._passiveReconstruct(T, b)
        
        return x_hat
    
    #Runs the passive reconstruction algorithm, taking bits b and generating
    #an estimate for the target vector using the measurement matrix self.M
    def _passiveReconstruct(self, T, b):
        a = 1/T*(self.M.T)@b;
        gamma0 = 2*self.c*np.sqrt((self.delta+np.log(self.d))/T);
        if np.max(a) <= gamma0:
            x_hat = np.zeros([self.d, 1])
        else:
            a = self._softThresholding(a, gamma0)
            x_hat = a/np.linalg.norm(a, 2)
        return x_hat
        
    def _softThresholding(self, alpha, gamma0):
        #SOFT_THRESHOLDING Equation (7) in Zhang2014
        # alpha = real-valued vector of any length (including 1)
        # gamma = real-valued scalar
        return np.where( np.abs(alpha) > gamma0, np.sign(alpha)*(np.abs(alpha) - gamma0), 0 );
        
    #Takes a matrix of tx values, where each column corresponds to one device
    #Returns a vector that corresponds to the output of the MAC
    def _MAC(self, tx_matrix, l):
        W = tx_matrix
        T = W.shape[0]
        n = W.shape[1]
        H = np.random.randn(T, n) + 1j*np.random.randn(T, n)
        H = self.sigma_h*H
        z = np.random.randn(T, 1) + 1j*np.random.randn(T, 1)
        z = self.sigma_z*z
        
        return np.matmul( np.multiply(W, H), l ) + z
        
        
    #GENERATE_PATHLOSS Generates pathloss according to uniform circular
    #cell model.
    #INPUT:
    # n = number of devices (positive integer)
    # R = cell radius (positive scalar)
    # beta = path loss exponent (positive scalar)
    #OUTPUT:
    # l = path losses (n-length vector)
    def _generate_pathloss(self, n, R, beta):
        r = np.zeros([n, 1]);
        for i in range(n):
            tri = 2 # just some number bigger than 1
            # The distance is a right triangular distribution
            while tri > 1:
                r_a = np.random.rand(1);
                r_b = np.random.rand(1);
                # Sum of two uniform distributions is triangle distribution
                tri = r_a+r_b;
            
            r[i] = tri*R;
            
        l = np.power(r, -beta/2)
        return l


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        