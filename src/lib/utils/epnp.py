# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 13:11:30 2017

@author: Weiyan Cai
"""

import math
import numpy as np
import utils.constraintMatrix as cM
import utils.gaussOptimization as gOpt


THRESHOLD_REPROJECTION_ERROR = 10


class EPnP(object):
    def __init__(self):
        self.Cw = self.define_control_points()
        
    def handle_general_EPnP(self, Xworld, Ximg_pix, A):
        self.n = len(Ximg_pix)
        self.A = A
        self.Alpha = self.compute_alphas(Xworld)
        print(self.Alpha)
        M = self.compute_M_ver2(Ximg_pix)
        self.K = self.kernel_noise(M, 4)#特征向量

        K = self.K
        kernel = np.array([K.T[3], K.T[2], K.T[1], K.T[0]]).T
        L6_10 = cM.compute_L6_10(kernel)
        error, Rt, Cc, Xc, sc, beta = self.dim_kerM(K[:, 0], Xworld, Ximg_pix, L6_10)
        print('Xc')
        print(Xc)
        Urep = np.matmul(A[:, :3], Xc.T).T
        Urep[:, 0] = Urep[:, 0] / Urep[:, 2]
        Urep[:, 1] = Urep[:, 1] / Urep[:, 2]
        new_point2d = Urep[:, 0:2]
        print("result 2D:")
        print(Urep[:, 0:2])
        # for i in range(2):
        #     error, Rt, Cc, Xc, sc, beta = self.dim_kerM(i + 1, K[:, :(i + 1)], Xworld, Ximg_pix, L6_10)
        #     errors.append(error)
        #     Rt_sol.append(Rt)
        #     Cc_sol.append(Cc)
        #     Xc_sol.append(Xc)
        #     sc_sol.append(sc)
        #     beta_sol.append(beta)
        #
        # if min(errors) > THRESHOLD_REPROJECTION_ERROR:
        #     error, Rt, Cc, Xc, sc, beta = self.dim_kerM(3, K[:, :3], Xworld, Ximg_pix, L6_10)
        #     errors.append(error)
        #     Rt_sol.append(Rt)
        #     Cc_sol.append(Cc)
        #     Xc_sol.append(Xc)
        #     sc_sol.append(sc)
        #     beta_sol.append(beta)
        #
        # best = np.array(errors).argsort()[0]
        # error_best = errors[best]
        # Rt_best, Cc_best, Xc_best = Rt_sol[best], Cc_sol[best], Xc_sol[best]
        # sc_best, beta_best = sc_sol[best], beta_sol[best]
        # print('Xc')
        # print(Xc_sol)
        return error, Rt, Cc, Xc, sc, beta, new_point2d
    
    def efficient_pnp(self, Xworld, Ximg_pix, A):
        error_best, Rt_best, Cc_best, Xc_best, _, _, new_point2d = self.handle_general_EPnP(Xworld, Ximg_pix, A)
        return error_best, Rt_best, Cc_best, Xc_best, new_point2d
        
    # def efficient_pnp_gauss(self, Xworld, Ximg_pix, A):
    #     error_best, Rt_best, Cc_best, Xc_best, sc_best, beta_best = self.handle_general_EPnP(Xworld, Ximg_pix, A)
    #
    #     best = len(beta_best)
    #     if best == 1:
    #         Betas = [0, 0, 0, beta_best[0]]
    #     elif best == 2:
    #         Betas = [0, 0, beta_best[0], beta_best[1]]
    #     else:
    #         Betas = [0, beta_best[0], beta_best[1], beta_best[2]]
    #
    #     Beta0 = sc_best * np.array(Betas)
    #     K = self.K
    #     Kernel = np.array([K.T[3], K.T[2], K.T[1], K.T[0]]).T
    #
    #     Xc_opt, Cc_opt, Rt_opt, err_opt = self.optimize_betas_gauss_newton(Kernel, Beta0, Xworld, Ximg_pix)
    #
    #     if err_opt < error_best:
    #         error_best, Rt_best, Cc_best, Xc_best = err_opt, Rt_opt, Cc_opt, Xc_opt
    #
    #     return error_best, Rt_best, Cc_best, Xc_best
        
    def optimize_betas_gauss_newton(self, Kernel, Beta0, Xw, U):
        n = len(Beta0)
        Beta_opt, error_opt = gOpt.gauss_newton(Kernel, self.Cw, Beta0)
        X = np.zeros((12))
        for i in range(n):
            X = X + Beta_opt[i] * Kernel[:, i]
        
        Cc = []
        for i in range(4):
            Cc.append(X[(3 * i) : (3 * (i + 1))])
        
        Cc = np.array(Cc).reshape((4, 3))
        s_Cw = gOpt.sign_determinant(self.Cw)
        s_Cc = gOpt.sign_determinant(Cc)
        Cc = Cc * s_Cw * s_Cc
        
        Xc_opt = np.matmul(self.Alpha, Cc)
        R_opt, T_opt = self.getRotT(Xw, Xc_opt)
        Rt_opt = np.concatenate((R_opt.reshape((3, 3)), T_opt.reshape((3, 1))), axis=1)
        err_opt = self.reprojection_error_usingRT(Xw, U, Rt_opt)
        
        return Xc_opt, Cc, Rt_opt, err_opt

    def define_control_points(self):
        return np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        
    def compute_alphas(self, Xworld):
        X = np.array(Xworld).transpose()
        X = np.concatenate((X, np.array([np.ones((self.n))])), axis=0)
        C = self.Cw.transpose()
        C = np.concatenate((C, np.array([np.ones((4))])), axis=0)
        
        Alpha = np.matmul(np.linalg.inv(C), X)
        Alpha = Alpha.transpose()
        return Alpha
    
    def compute_M_ver2(self, U):
        A, Alpha = self.A, self.Alpha
        fu, fv, u0, v0 = A[0, 0], A[1, 1], A[0, 2], A[1, 2]
        print(fu, fv, u0, v0)
        M = np.array([[0.0] * 12] * 18)
        U = np.array(U)
        print(U)
        for i in range(self.n):
            uci = u0 + U[i, 0]
            vci = v0 + U[i, 1]
            for j in range(4):
                barycentric = Alpha[i, j]
                M[2 * i, 3 * j] = barycentric * fu
                M[2 * i, 3 * j + 2] = barycentric * uci
                M[2 * i + 1, 3 * j + 1] = barycentric * fv
                M[2 * i + 1, 3 * j + 2] = barycentric * vci

        return M
    
    def kernel_noise(self, M, dimker):
        M = np.array(M)
        M_T_M = np.matmul(M.transpose(), M)
        W, V = np.linalg.eig(M_T_M)
        idx = W.argsort()
        K = V[:, idx[:dimker]]
        return K 
    
    def dim_kerM(self, Km, Xworld, Ximg_pix, L6_10):

        X1 = Km
        Cc, Xc, sc = self.compute_norm_sign_scaling_factor(X1, Xworld)
        beta = [1]
           
        # if dimker == 2:
        #     L = cM.compute_L6_3(L6_10)
        #     dsp = cM.compute_rho(self.Cw)
        #     betas = np.matmul(np.linalg.inv(np.matmul(L.T, L)), np.matmul(L.T, dsp))#np.linalg.inv()：矩阵求逆
        #     beta1 = math.sqrt(abs(betas[0]))
        #     beta2 = math.sqrt(abs(betas[2])) * np.sign(betas[1]) * np.sign(betas[0])
        #
        #     X2 = beta1 * Km.T[1] + beta2 * Km.T[0]
        #     Cc, Xc, sc = self.compute_norm_sign_scaling_factor(X2, Xworld)
        #     beta = [beta1, beta2]
        #
        # if dimker == 3:
        #     L = cM.compute_L6_6(L6_10)
        #     dsp = cM.compute_rho(self.Cw)
        #     betas = np.matmul(np.linalg.inv(L), dsp)
        #     beta1 = math.sqrt(abs(betas[0]))
        #     beta2 = math.sqrt(abs(betas[3])) * np.sign(betas[1]) * np.sign(betas[0])
        #     beta3 = math.sqrt(abs(betas[5])) * np.sign(betas[2]) * np.sign(betas[0])
        #     print('Km')
        #     print(Km)
        #     X3 = beta1 * Km.T[2] + beta2 * Km.T[1] + beta3 * Km.T[0]
        #     # print('X3')
        #     # print(X3)
        #     Cc, Xc, sc = self.compute_norm_sign_scaling_factor(X3, Xworld)
        #     # print('Xc')
        #     # print(Xc)
        #     beta = [beta1, beta2, beta3]
            
        R, T = self.getRotT(Xworld, Xc)
        Rt = np.concatenate((R.reshape((3, 3)), T.reshape((3, 1))), axis=1)
        error = self.reprojection_error_usingRT(Xworld, Ximg_pix, Rt)
        #error = self.reprojection_error_usingRT(Xc, Ximg_pix, Rt)
        return error, Rt, Cc, Xc, sc, beta
            
    def compute_norm_sign_scaling_factor(self, X, Xworld):
        Cc = []
    
        for i in range(4):
            Cc.append(X[(3 * i) : (3 * (i + 1))])
        Cc = np.array(Cc).reshape((4, 3))
        if Cc[0, 2] > 0:
            Cc *= -1
        Xc = np.matmul(self.Alpha, Cc)

        # centr_w = np.mean(Xworld, axis=0)
        # centroid_w = np.tile(centr_w.reshape((1, 3)), (self.n, 1))#np.tile 平铺
        # tmp1 = Xworld.reshape((self.n, 3)) - centroid_w
        # dist_w = np.sqrt(np.sum(tmp1 ** 2, axis=1))
        #
        # centr_c = np.mean(np.array(Xc), axis=0)
        # centroid_c = np.tile(centr_c.reshape((1, 3)), (self.n, 1))
        # tmp2 = Xc.reshape((self.n, 3)) - centroid_c
        # dist_c = np.sqrt(np.sum(tmp2 ** 2, axis=1))
        #
        # sc_1 = np.matmul(dist_c.transpose(), dist_c) ** -1
        # sc_2 = np.matmul(dist_c.transpose(), dist_w)
        # sc = sc_1 * sc_2
        # print("sc")
        # print(sc)
        # Cc *= sc
        # Xc = np.matmul(self.Alpha, Cc)
        
#         for x in Xc:
#             if x[-1] < 0:
#                 Xc *= -1
#                 Cc *= -1
        return Cc, Xc, 1.0
    
    def getRotT(self, wpts, cpts):
        wcent = np.tile(np.mean(wpts, axis=0).reshape((1, 3)), (self.n, 1))
        ccent = np.tile(np.mean(cpts, axis=0).reshape((1, 3)), (self.n, 1))
        
        wpts = wpts.reshape((self.n, 3)) - wcent
        cpts = cpts.reshape((self.n, 3)) - ccent
        
        M = np.matmul(cpts.transpose(), wpts)
        
        U, S, V = np.linalg.svd(M)
        R = np.matmul(U, V)
        
        if np.linalg.det(R) < 0:
            R = - R
            
        T = ccent[0].transpose() - np.matmul(R, wcent[0].transpose())
        
        return R, T   
    
    def reprojection_error_usingRT(self, Xw, U, RT):
        A = self.A
        P = np.matmul(A[:, :3], RT)
        Xw_h = np.concatenate((Xw.reshape((self.n, 3)), np.array([np.ones((self.n))]).T), axis=1)
    
        Urep = np.matmul(P, Xw_h.T).T
        Urep[:, 0] = Urep[:, 0] / Urep[:, 2]
        Urep[:, 1] = Urep[:, 1] / Urep[:, 2]
        print("U")
        print(U)
        err = np.sqrt((U[:, 0] - Urep[:, 0].reshape((self.n, 1))) ** 2 + (U[:, 1] - Urep[:, 1].reshape((self.n, 1))) ** 2)
        err = np.sum(err, axis=0) / self.n

        return err[0]
        

if __name__ == "__main__":
    EPnP = EPnP()
