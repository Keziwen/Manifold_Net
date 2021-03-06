import tensorflow as tf
from tensorflow.keras import layers
import os
import numpy as np
import time
from tools.tools import tempfft, fft2c_mri, ifft2c_mri, Emat_xyt


class CNNLayer(tf.keras.layers.Layer):
    def __init__(self, n_f=32, n_out=2):
        super(CNNLayer, self).__init__()
        self.mylayers = []

        self.mylayers.append(tf.keras.layers.Conv3D(n_f, 3, strides=1, padding='same', use_bias=False))
        self.mylayers.append(tf.keras.layers.LeakyReLU())
        self.mylayers.append(tf.keras.layers.Conv3D(n_f, 3, strides=1, padding='same', use_bias=False))
        self.mylayers.append(tf.keras.layers.LeakyReLU())
        self.mylayers.append(tf.keras.layers.Conv3D(n_f, 3, strides=1, padding='same', use_bias=False))
        self.mylayers.append(tf.keras.layers.LeakyReLU())
        self.mylayers.append(tf.keras.layers.Conv3D(n_f, 3, strides=1, padding='same', use_bias=False))
        self.mylayers.append(tf.keras.layers.LeakyReLU())
        self.mylayers.append(tf.keras.layers.Conv3D(n_out, 3, strides=1, padding='same', use_bias=False))
        self.seq = tf.keras.Sequential(self.mylayers)

    def call(self, input):
        if len(input.shape) == 4:
            input2c = tf.stack([tf.math.real(input), tf.math.imag(input)], axis=-1)
        else:
            input2c = tf.concat([tf.math.real(input), tf.math.imag(input)], axis=-1)
        res = self.seq(input2c)
        res = tf.complex(res[:,:,:,:,0], res[:,:,:,:,1])
        
        return res

class CONV_OP(tf.keras.layers.Layer):
    def __init__(self, n_f=32, ifactivate=False):
        super(CONV_OP, self).__init__()
        self.mylayers = []
        self.mylayers.append(tf.keras.layers.Conv3D(n_f, 3, strides=1, padding='same', use_bias=False))
        if ifactivate == True:
            self.mylayers.append(tf.keras.layers.ReLU())
        self.seq = tf.keras.Sequential(self.mylayers)

    def call(self, input):
        res = self.seq(input)
        return res

class SLR_Net(tf.keras.Model):
    def __init__(self, mask, niter, learned_topk=False):
        super(SLR_Net, self).__init__(name='SLR_Net')
        self.niter = niter
        self.E = Emat_xyt(mask)
        self.learned_topk = learned_topk
        self.celllist = []
    

    def build(self, input_shape):
        for i in range(self.niter-1):
            self.celllist.append(SLRCell(input_shape, self.E, learned_topk=self.learned_topk))
        self.celllist.append(SLRCell(input_shape, self.E, learned_topk=self.learned_topk, is_last=True))

    def call(self, d, csm):
        """
        d: undersampled k-space
        csm: coil sensitivity map
        """
        if csm == None:
            nb, nt, nx, ny = d.shape
        else:
            nb, nc, nt, nx, ny = d.shape
        X_SYM = []
        x_rec = self.E.mtimes(d, inv=True, csm=csm)
        t = tf.zeros_like(x_rec)
        beta = tf.zeros_like(x_rec)
        x_sym = tf.zeros_like(x_rec)
        data = [x_rec, x_sym, beta, t, d, csm]
        
        for i in range(self.niter):
            data = self.celllist[i](data, d.shape)
            x_sym = data[1]
            X_SYM.append(x_sym)

        x_rec = data[0]
        
        return x_rec, X_SYM


class SLRCell(layers.Layer):
    def __init__(self, input_shape, E, learned_topk=False, is_last=False):
        super(SLRCell, self).__init__()
        if len(input_shape) == 4:
            self.nb, self.nt, self.nx, self.ny = input_shape
        else:
            self.nb, nc, self.nt, self.nx, self.ny = input_shape

        self.E = E
        self.learned_topk = learned_topk
        if self.learned_topk:
            if is_last:
                self.thres_coef = tf.Variable(tf.constant(-2, dtype=tf.float32), trainable=False, name='thres_coef')
                self.eta = tf.Variable(tf.constant(0.01, dtype=tf.float32), trainable=False, name='eta')
            else:
                self.thres_coef = tf.Variable(tf.constant(-2, dtype=tf.float32), trainable=True, name='thres_coef')
                self.eta = tf.Variable(tf.constant(0.01, dtype=tf.float32), trainable=True, name='eta')

        self.conv_1 = CONV_OP(n_f=16, ifactivate=True)
        self.conv_2 = CONV_OP(n_f=16, ifactivate=True)
        self.conv_3 = CONV_OP(n_f=16, ifactivate=False)
        self.conv_4 = CONV_OP(n_f=16, ifactivate=True)
        self.conv_5 = CONV_OP(n_f=16, ifactivate=True)
        self.conv_6 = CONV_OP(n_f=2, ifactivate=False)
        #self.conv_7 = CONV_OP(n_f=16, ifactivate=True)
        #self.conv_8 = CONV_OP(n_f=16, ifactivate=True)
        #self.conv_9 = CONV_OP(n_f=16, ifactivate=True)
        #self.conv_10 = CONV_OP(n_f=16, ifactivate=True)

        self.lambda_step = tf.Variable(tf.constant(0.1, dtype=tf.float32), trainable=True, name='lambda_1')
        self.lambda_step_2 = tf.Variable(tf.constant(0.1, dtype=tf.float32), trainable=True, name='lambda_2')
        self.soft_thr = tf.Variable(tf.constant(0.1, dtype=tf.float32), trainable=True, name='soft_thr')


    def call(self, data, input_shape):
        if len(input_shape) == 4:
            self.nb, self.nt, self.nx, self.ny = input_shape
        else:
            self.nb, nc, self.nt, self.nx, self.ny = input_shape
        x_rec, x_sym, beta, t, d, csm = data

        
        x_rec, x_sym = self.sparse(x_rec, d, t, beta, csm)
        t = self.lowrank(x_rec)
        
        beta = self.beta_mid(beta, x_rec, t)

        data[0] = x_rec
        data[1] = x_sym
        data[2] = beta
        data[3] = t

        return data

    def sparse(self, x_rec, d, t, beta, csm):
        lambda_step = tf.cast(tf.nn.relu(self.lambda_step), tf.complex64)
        lambda_step_2 = tf.cast(tf.nn.relu(self.lambda_step_2), tf.complex64)

        ATAX_cplx = self.E.mtimes(self.E.mtimes(x_rec, inv=False, csm=csm) - d, inv=True, csm=csm)

        r_n = x_rec - tf.math.scalar_mul(lambda_step, ATAX_cplx) +\
              tf.math.scalar_mul(lambda_step_2, x_rec + beta - t)

        # D_T(soft(D_r_n))
        if len(r_n.shape) == 4:
            r_n = tf.stack([tf.math.real(r_n), tf.math.imag(r_n)], axis=-1)

        x_1 = self.conv_1(r_n)
        x_2 = self.conv_2(x_1)
        x_3 = self.conv_3(x_2)

        x_soft = tf.math.multiply(tf.math.sign(x_3), tf.nn.relu(tf.abs(x_3) - self.soft_thr))

        x_4 = self.conv_4(x_soft)
        x_5 = self.conv_5(x_4)
        x_6 = self.conv_6(x_5)

        x_rec = x_6 + r_n

        x_1_sym = self.conv_4(x_3)
        x_1_sym = self.conv_5(x_1_sym)
        x_1_sym = self.conv_6(x_1_sym)
        #x_sym_1 = self.conv_10(x_1_sym)

        x_sym = x_1_sym - r_n
        x_rec = tf.complex(x_rec[:, :, :, :, 0], x_rec[:, :, :, :, 1])

        return x_rec, x_sym

    def lowrank(self, x_rec):
        [batch, Nt, Nx, Ny] = x_rec.get_shape()
        M = tf.reshape(x_rec, [batch, Nt, Nx*Ny])
        St, Ut, Vt = tf.linalg.svd(M)
        if self.learned_topk:
            #tf.print(tf.sigmoid(self.thres_coef))
            thres = tf.sigmoid(self.thres_coef) * St[:, 0]
            thres = tf.expand_dims(thres, -1)
            St = tf.nn.relu(St - thres)
        else:
            top1_mask = np.concatenate(
                [np.ones([self.nb, 1], dtype=np.float32), np.zeros([self.nb, self.nt - 1], dtype=np.float32)], 1)
            top1_mask = tf.constant(top1_mask)
            St = St * top1_mask
        St = tf.linalg.diag(St)
        
        St = tf.dtypes.cast(St, tf.complex64)
        Vt_conj = tf.transpose(Vt, perm=[0, 2, 1])
        Vt_conj = tf.math.conj(Vt_conj)
        US = tf.linalg.matmul(Ut, St)
        M = tf.linalg.matmul(US, Vt_conj)
        x_rec = tf.reshape(M, [batch, Nt, Nx, Ny])

        return x_rec

    def beta_mid(self, beta, x_rec, t):
        eta = tf.cast(tf.nn.relu(self.eta), tf.complex64)
        return beta + tf.multiply(eta, x_rec - t)

###### DC-CNN ######
class DC_CNN_LR(tf.keras.Model):
    def __init__(self, mask, niter, learned_topk=False):
        super(DC_CNN_LR, self).__init__(name='DC_CNN_LR')
        self.niter = niter
        self.E = Emat_xyt(mask)
        self.mask = mask
        self.learned_topk = learned_topk
        self.celllist = []
    

    def build(self, input_shape):
        for i in range(self.niter-1):
            self.celllist.append(DNCell(input_shape, self.E, self.mask, learned_topk=self.learned_topk))
        self.celllist.append(DNCell(input_shape, self.E, self.mask, learned_topk=self.learned_topk, is_last=True))

    def call(self, d, csm):
        """
        d: undersampled k-space
        csm: coil sensitivity map
        """
        if csm == None:
            nb, nt, nx, ny = d.shape
        else:
            nb, nc, nt, nx, ny = d.shape
    
        x_rec = self.E.mtimes(d, inv=True, csm=csm)
        
        for i in range(self.niter):
            x_rec = self.celllist[i](x_rec, d, d.shape)
    
        return x_rec


class DNCell(layers.Layer):

    def __init__(self, input_shape, E, mask, learned_topk=False, is_last=False):
        super(DNCell, self).__init__()
        if len(input_shape) == 4:
            self.nb, self.nt, self.nx, self.ny = input_shape
        else:
            self.nb, nc, self.nt, self.nx, self.ny = input_shape

        self.E = E
        self.mask = mask
        self.learned_topk = learned_topk
        if self.learned_topk:
            if is_last:
                self.thres_coef = tf.Variable(tf.constant(-2, dtype=tf.float32), trainable=False, name='thres_coef')
                
            else:
                self.thres_coef = tf.Variable(tf.constant(-2, dtype=tf.float32), trainable=True, name='thres_coef')
                

        self.conv_1 = CONV_OP(n_f=16, ifactivate=True)
        self.conv_2 = CONV_OP(n_f=16, ifactivate=True)
        self.conv_3 = CONV_OP(n_f=16, ifactivate=True)
        self.conv_4 = CONV_OP(n_f=16, ifactivate=True)
        self.conv_5 = CONV_OP(n_f=2, ifactivate=False)

    def call(self, x_rec, d, input_shape):
        if len(input_shape) == 4:
            self.nb, self.nt, self.nx, self.ny = input_shape
        else:
            self.nb, nc, self.nt, self.nx, self.ny = input_shape
            
        x_rec = self.sparse(x_rec, d) 
  
        return x_rec

    def sparse(self, x_rec, d):

        
        r_n = tf.stack([tf.math.real(x_rec), tf.math.imag(x_rec)], axis=-1)

        x_1 = self.conv_1(r_n)
        x_2 = self.conv_2(x_1)
        x_3 = self.conv_3(x_2)
        x_4 = self.conv_4(x_3)
        x_5 = self.conv_5(x_4)
     
        x_rec = x_5 + r_n
        x_rec = tf.complex(x_rec[:, :, :, :, 0], x_rec[:, :, :, :, 1])
        if self.learned_topk:
            x_rec = self.lowrank(x_rec)
        x_rec = self.dc_layer(x_rec, d)

        return x_rec

    def lowrank(self, x_rec):
        [batch, Nt, Nx, Ny] = x_rec.get_shape()
        M = tf.reshape(x_rec, [batch, Nt, Nx*Ny])
        St, Ut, Vt = tf.linalg.svd(M)
        if self.learned_topk:
            #tf.print(tf.sigmoid(self.thres_coef))
            thres = tf.sigmoid(self.thres_coef) * St[:, 0]
            thres = tf.expand_dims(thres, -1)
            St = tf.nn.relu(St - thres)
        else:
            top1_mask = np.concatenate(
                [np.ones([self.nb, 1], dtype=np.float32), np.zeros([self.nb, self.nt - 1], dtype=np.float32)], 1)
            top1_mask = tf.constant(top1_mask)
            St = St * top1_mask
        St = tf.linalg.diag(St)
        
        St = tf.dtypes.cast(St, tf.complex64)
        Vt_conj = tf.transpose(Vt, perm=[0, 2, 1])
        Vt_conj = tf.math.conj(Vt_conj)
        US = tf.linalg.matmul(Ut, St)
        M = tf.linalg.matmul(US, Vt_conj)
        x_rec = tf.reshape(M, [batch, Nt, Nx, Ny])
        return x_rec
    
    def dc_layer(self, x_rec, d):

        k_rec = fft2c_mri(x_rec)
        k_rec = (1 - self.mask) * k_rec + self.mask * d
        x_rec = ifft2c_mri(k_rec)

        return x_rec
        
###### Manifold_Net ######
class Manifold_Net(tf.keras.Model):
    def __init__(self, mask, niter, learned_topk=False, N_factor=1):
        super(Manifold_Net, self).__init__(name='Manifold_Net')
        self.niter = niter
        self.E = Emat_xyt(mask)
        self.mask = mask
        self.learned_topk = learned_topk
        self.N_factor = N_factor
        self.celllist = []
    

    def build(self, input_shape):
        for i in range(self.niter-1):
            self.celllist.append(ManifoldCell(input_shape, self.E, self.mask, learned_topk=self.learned_topk, N_factor=self.N_factor))
        self.celllist.append(ManifoldCell(input_shape, self.E, self.mask, learned_topk=self.learned_topk, N_factor=self.N_factor, is_last=True))

    def call(self, d, csm):
        """
        d: undersampled k-space
        csm: coil sensitivity map
        """
        if csm == None:
            nb, nt, nx, ny = d.shape
        else:
            nb, nc, nt, nx, ny = d.shape
    
        x_rec = self.E.mtimes(d, inv=True, csm=csm)
        
        for i in range(self.niter):
            x_rec = self.celllist[i](x_rec, d, d.shape)
    
        return x_rec

class ManifoldCell(layers.Layer):

    def __init__(self, input_shape, E, mask, learned_topk=False, N_factor=1, is_last=False):
        super(ManifoldCell, self).__init__()
        if len(input_shape) == 4:
            self.nb, self.nt, self.nx, self.ny = input_shape
        else:
            self.nb, nc, self.nt, self.nx, self.ny = input_shape

        self.E = E
        self.mask = mask
        self.Nx_factor = N_factor
        self.Ny_factor = N_factor
        self.Nt_factor = N_factor
        self.learned_topk = learned_topk
        if self.learned_topk:
            self.eta = tf.Variable(tf.constant(0.01, dtype=tf.float32), trainable=True, name='eta')
            #self.lambda_sparse = tf.Variable(tf.constant(0.01, dtype=tf.float32), trainable=True, name='lambda')
                

        self.conv_1 = CNNLayer(n_f=16, n_out=2)
        #self.conv_2 = CNNLayer(n_f=16, n_out=2)
        #self.conv_3 = CNNLayer(n_f=16, n_out=2)
        #self.conv_D = CNNLayer(n_f=16, n_out=2)
        #self.conv_transD = CNNLayer(n_f=16, n_out=2)
       

    def call(self, x_rec, d, input_shape):
        if len(input_shape) == 4:
            self.nb, self.nt, self.nx, self.ny = input_shape
        else:
            self.nb, nc, self.nt, self.nx, self.ny = input_shape
        
        x_k = self.conv_1(x_rec)
        #grad_sparse = self.conv_transD(self.conv_D(x_k))
        #grad_sparse = tf.stack([tf.math.real(grad_sparse), tf.math.imag(grad_sparse)], axis=-1)
        #grad_sparse = tf.multiply(self.lambda_sparse, grad_sparse)
        #grad_sparse = tf.complex(grad_sparse[..., 0], grad_sparse[..., 1])
        #grad_dc = ifft2c_mri((fft2c_mri(x_k) * self.mask - d) * self.mask) 
        #g_k = grad_dc +  grad_sparse

        g_k = ifft2c_mri((fft2c_mri(x_k) * self.mask - d) * self.mask)
        #g_k = self.E.mtimes(self.E.mtimes(x_k, inv=False, csm=csm) - d, inv=True, csm=csm)
        t_k = self.Tangent_Module(g_k, x_k)
        x_k = self.Retraction_Module(x_k, t_k)
        #x_k = self.conv_3(x_k)

        x_k = self.dc_layer(x_k, d)
    
        return x_k

    def Tangent_Module(self, g_k, x_k):
        batch, Nt, Nx, Ny = x_k.shape
        x_k = tf.transpose(x_k, [0, 2, 3, 1]) # batch, Nx, Ny, Nt
        g_k = tf.transpose(g_k, [0, 2, 3, 1]) # batch, Nx, Ny, Nt

        Ux, Uy, Ut = self.Mode(x_k)
        first_term = self.Mode_Multiply(g_k, tf.transpose(Ux, [0, 2, 1], conjugate=True), mode_n=1)
        first_term = self.Mode_Multiply(first_term, tf.transpose(Uy, [0, 2, 1], conjugate=True), mode_n=2)
        first_term = self.Mode_Multiply(first_term, tf.transpose(Ut, [0, 2, 1], conjugate=True), mode_n=3)
        first_term = self.Mode_Multiply(first_term, Ux, mode_n=1)
        first_term = self.Mode_Multiply(first_term, Uy, mode_n=2)
        first_term = self.Mode_Multiply(first_term, Ut, mode_n=3)

        C_mode_x, C_mode_y, C_mode_t = self.Core_C(x_k, Ux, Uy, Ut)

        second_term_1 = self.Mode_Multiply(g_k, tf.transpose(Uy, [0, 2, 1], conjugate=True), mode_n=2)
        second_term_1 = self.Mode_Multiply(second_term_1, tf.transpose(Ut, [0, 2, 1], conjugate=True), mode_n=3)
        second_term_1 = tf.reshape(second_term_1, [batch, Nx, Ny * Nt])
        second_term_1 = self.Projector(second_term_1, Ux)
        second_term_1 = self.Core_Multiply(second_term_1, C_mode_x)
        second_term_1 = tf.linalg.matmul(second_term_1, C_mode_x)
        second_term_1 = tf.reshape(second_term_1, [batch, Nx, Ny, Nt])
        second_term_1 = self.Mode_Multiply(second_term_1, Uy, mode_n=2)
        second_term_1 = self.Mode_Multiply(second_term_1, Ut, mode_n=3)

        second_term_2 = self.Mode_Multiply(g_k, tf.transpose(Ux, [0, 2, 1], conjugate=True), mode_n=1)
        second_term_2 = self.Mode_Multiply(second_term_2, tf.transpose(Ut, [0, 2, 1], conjugate=True), mode_n=3)
        second_term_2 = tf.reshape(tf.transpose(second_term_2, [0, 2, 1, 3]), [batch, Ny, Nx*Nt])
        second_term_2 = self.Projector(second_term_2, Uy)
        second_term_2 = self.Core_Multiply(second_term_2, C_mode_y)
        second_term_2 = tf.linalg.matmul(second_term_2, C_mode_y)
        second_term_2 = tf.transpose(tf.reshape(second_term_2, [batch, Ny, Nx, Nt]), [0, 2, 1, 3])
        second_term_2 = self.Mode_Multiply(second_term_2, Ux, mode_n=1)
        second_term_2 = self.Mode_Multiply(second_term_2, Ut, mode_n=3)

        second_term_3 = self.Mode_Multiply(g_k, tf.transpose(Ux, [0, 2, 1], conjugate=True), mode_n=1)
        second_term_3 = self.Mode_Multiply(second_term_3, tf.transpose(Uy, [0, 2, 1], conjugate=True), mode_n=2)
        second_term_3 = tf.reshape(tf.transpose(second_term_3, [0, 3, 1, 2]), [batch, Nt, Nx * Ny])
        second_term_3 = self.Projector(second_term_3, Ut)
        second_term_3 = self.Core_Multiply(second_term_3, C_mode_t)
        second_term_3 = tf.linalg.matmul(second_term_3, C_mode_t)
        second_term_3 = tf.transpose(tf.reshape(second_term_3, [batch, Nt, Nx, Ny]), [0, 2, 3, 1])
        second_term_3 = self.Mode_Multiply(second_term_3, Ux, mode_n=1)
        second_term_3 = self.Mode_Multiply(second_term_3, Uy, mode_n=2)

        t_k = first_term + second_term_1 + second_term_2 + second_term_3
        t_k = tf.transpose(t_k, [0, 3, 1, 2])
        
        return t_k
    
    def Retraction_Module(self, x_k, t_k):
        x_k = tf.stack([tf.math.real(x_k), tf.math.imag(x_k)], axis=-1)
        t_k = tf.stack([tf.math.real(t_k), tf.math.imag(t_k)], axis=-1)
        x_k = x_k - tf.multiply(self.eta, t_k)
        x_k = tf.complex(x_k[..., 0], x_k[..., 1])

        batch, Nt, Nx, Ny = x_k.shape
        x_k = tf.transpose(x_k, [0, 2, 3, 1]) # batch, Nx, Ny, Nt

        Ux, Uy, Ut = self.Mode(x_k)
        Ux = self.SVT_U(Ux, top_kth= int(Nx / self.Nx_factor))
        Uy = self.SVT_U(Uy, top_kth= int(Ny / self.Ny_factor))
        Ut = self.SVT_U(Ut, top_kth= int(Nt / self.Nt_factor))
        """
        Ux = self.SVT_U(Ux, top_kth= Nx // self.Nx_factor)
        Uy = self.SVT_U(Uy, top_kth= Ny // self.Ny_factor)
        Ut = self.SVT_U(Ut, top_kth= Nt // self.Nt_factor)
        """
        C = self.Mode_Multiply(x_k, tf.transpose(Ux, [0, 2, 1], conjugate=True), mode_n=1)
        C = self.Mode_Multiply(C, tf.transpose(Uy, [0, 2, 1], conjugate=True), mode_n=2)
        C = self.Mode_Multiply(C, tf.transpose(Ut, [0, 2, 1], conjugate=True), mode_n=3)

        x_k = self.Mode_Multiply(C, Ux, mode_n=1)
        x_k = self.Mode_Multiply(x_k, Uy, mode_n=2)
        x_k = self.Mode_Multiply(x_k, Ut, mode_n=3)

        x_k = tf.transpose(x_k, [0, 3, 1, 2])

        return x_k
    
    def Mode(self, x_k):
        batch, Nx, Ny, Nt = x_k.shape
        
        mode_x = tf.reshape(x_k, [batch, Nx, Ny*Nt])
        mode_y = tf.reshape(tf.transpose(x_k, [0, 2, 1, 3]), [batch, Ny, Nx*Nt])
        mode_t = tf.reshape(tf.transpose(x_k, [0, 3, 1, 2]), [batch, Nt, Nx*Ny])

        Sx, Ux, Vx = tf.linalg.svd(mode_x) # Ux: batch, 192, 192
        Sy, Uy, Vy = tf.linalg.svd(mode_y) # Uy: batch, 192, 192
        St, Ut, Vt = tf.linalg.svd(mode_t)

        return Ux, Uy, Ut
    
    def Mode_Multiply(self, A, U, mode_n=1):
        """
        A: batch, Nx, Ny, Nt
        U: batch, Nx, Ny
        return: batch, Nx, Ny, Nt
        """
        batch, Nx, Ny, Nt = A.shape

        if mode_n == 1:
            out = tf.linalg.matmul(U, tf.reshape(A, [batch, Nx, Ny*Nt])) # batch, Nx, Ny*Nt
            out= tf.reshape(out, [batch, Nx, Ny, Nt])
        elif mode_n == 2:
            out = tf.linalg.matmul(U, tf.reshape(tf.transpose(A, [0, 2, 1, 3]), [batch, Ny, Nx * Nt]))  # batch, Ny, Nx*Nt
            out = tf.transpose(tf.reshape(out, [batch, Ny, Nx, Nt]), [0, 2, 1, 3])
        elif mode_n == 3:
            out = tf.linalg.matmul(U, tf.reshape(tf.transpose(A, [0, 3, 1, 2]), [batch, Nt, Nx * Ny]))  # batch, Nt, Nx*Ny
            out = tf.transpose(tf.reshape(out, [batch, Nt, Nx, Ny]), [0, 2, 3, 1])

        return out

    def Core_C(self, x_k, Ux, Uy, Ut):
        batch, Nx, Ny, Nt = x_k.shape

        C = self.Mode_Multiply(x_k, tf.transpose(Ux, [0, 2, 1], conjugate=True), mode_n=1)
        C = self.Mode_Multiply(C, tf.transpose(Uy, [0, 2, 1], conjugate=True), mode_n=2)
        C = self.Mode_Multiply(C, tf.transpose(Ut, [0, 2, 1], conjugate=True), mode_n=3)
        
        C_mode_x = tf.reshape(C, [batch, Nx, Ny * Nt])
        C_mode_y = tf.reshape(tf.transpose(C, [0, 2, 1, 3]), [batch, Ny, Nx * Nt])
        C_mode_t = tf.reshape(tf.transpose(C, [0, 3, 1, 2]), [batch, Nt, Nx * Ny])

        return C_mode_x, C_mode_y, C_mode_t

    def Projector(self, second_term, U):
        second_term = second_term - tf.linalg.matmul(
            tf.linalg.matmul(U, 
            tf.transpose(U, [0, 2, 1], conjugate=True)), 
            second_term)
        
        return second_term

    def Core_Multiply(self, second_term, C_mode):
        second_term = tf.linalg.matmul(second_term,
                                     tf.linalg.matmul(tf.transpose(C_mode, [0, 2, 1], conjugate=True),
                                                      tf.linalg.inv(tf.linalg.matmul(C_mode,
                                                                                     tf.transpose(C_mode, [0, 2, 1], conjugate=True)))))
        return second_term

    def SVT_U(self, Uk, top_kth):
        [batch, Nx, Ny] = Uk.get_shape()
        mask_1 = tf.ones([batch, Nx, top_kth])
        mask_2 = tf.zeros([batch, Nx, Ny - top_kth])
        mask_top_k = tf.concat([mask_1, mask_2], axis=-1)
        mask_top_k = tf.cast(mask_top_k, dtype=Uk.dtype)
        Uk = tf.multiply(Uk, mask_top_k)
        return Uk

    def dc_layer(self, x_rec, d):

        k_rec = fft2c_mri(x_rec)
        k_rec = (1 - self.mask) * k_rec + self.mask * d
        x_rec = ifft2c_mri(k_rec)

        return x_rec

    def dc_layer_v2(self, x_rec, d):
        x_rec = x_rec - ifft2c_mri(fft2c_mri(x_rec) * self.mask - d)
        return x_rec
    
        




