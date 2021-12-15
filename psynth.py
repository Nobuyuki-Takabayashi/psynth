# Creted on March 30, 2021

import numpy as np
import math
import pandas as pd
from scipy import constants
from scipy import signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import datetime
import sys

def P(n, r):
    return math.factorial(n)//math.factorial(n-r)

def C(n, r):
    return P(n, r)//math.factorial(r)

today_ = datetime.datetime.today()
today = today_.strftime('%y%m%d')

class PhasedArray(object):

    def __init__(self, frequency, element_num_x, element_num_y,element_interval_x, element_interval_y):
        # frequency [GHz], Operation Frequency on Array Antenna
        # elementNumX [None], Number of Element in X Direction
        # elementNumY [None], Number of Element in Y Direction
        # elementIntervalX [mm], Interval between Elements on X Direction
        # elementIntervalY [mm], Interval between Elements on Y Direction
        # Suppose a Square Arrangement as a Basic State.
        self.frequency = frequency # Unit [GHz]
        self.wavelength = constants.c / (self.frequency*10**9) # Unit [m]
        self.wavenumber = 2*np.pi/self.wavelength
        self.element_num_x = element_num_x
        self.element_num_y = element_num_y
        self.element_int_x = element_interval_x
        self.element_int_y = element_interval_y
        self.antenna_pos_x = np.arange(-(self.element_num_x-1)/2.0, (self.element_num_x+1)/2.0, 1.0)*element_interval_x/1000
        self.antenna_pos_y = np.arange(-(self.element_num_y-1)/2.0, (self.element_num_y+1)/2.0, 1.0)*element_interval_y/1000
        self.element_int_wl_x = element_interval_x/self.wavelength # Unit [None]
        self.element_int_wl_y = element_interval_y/self.wavelength # Unit [None]
        self.antenna_pos_x_mesh, self.antenna_pos_y_mesh = np.meshgrid(self.antenna_pos_x, self.antenna_pos_y)
        self.array_manner = "square"
        self.array_weight = np.ones((self.element_num_y,self.element_num_x),dtype='complex128')/ np.sqrt(self.element_num_y*self.element_num_x)
        self.excitation_manner = "uniform"

    ###############################
    ###### Array Arrangement ###### if necessary
    ###############################
    def align_square(self, element_int_wl_x=0.7, element_int_wl_y=0.7, x_num=14, y_num=14):
        self.element_num_x = x_num
        self.element_num_y = y_num
        self.element_int_wl_x = element_int_wl_x
        self.element_int_wl_y = element_int_wl_y
        self.element_int_x = pitch_lambda_x * self.wavelength
        self.element_int_y = pitch_lambda_y * self.wavelength
        self.element_num = (x_num, y_num)
        self.antenna_pos_x = np.arange(-(x_num-1)/2.0, (x_num+1)/2.0, 1.0)*self.element_int_x
        self.antenna_pos_y = np.arange(-(y_num-1)/2.0, (y_num+1)/2.0, 1.0)*self.element_int_y
        self.antenna_pos_x_mesh, self.antenna_pos_y_mesh = np.meshgrid(self.antenna_pos_x, self.antenna_pos_y)
        return self

    def align_circle(self, circle_radius, circle_element_num):
        self.circle_radius = circle_radius
        self.circle_element_num = circle_element_num
        dphi = 2*np.pi / element_num
        self.antenna_pos_x = np.cos(np.arange(0, 2*np.pi, dphi))*self.circle_radius
        self.antenna_pos_y = np.sin(np.arange(0, 2*np.pi, dphi))*self.circle_radius
        self.array_manner = "circle"
        return self

    def array_delete(self, xlist, ylist):
        x_temp = np.delete(self.antenna_pos_x,xlist)
        y_temp = np.delete(self.antenna_pos_y,ylist)
        x_num_temp = self.element_num_x - len(xlist)
        y_num_temp = self.element_num_y - len(ylist)
        self.element_num_x = x_num_temp
        self.element_num_y = y_num_temp
        self.antenna_pos_x = x_temp
        self.antenna_pos_y = y_temp
        self.antenna_pos_x_mesh, self.antenna_pos_y_mesh = np.meshgrid(self.antenna_pos_x, self.antenna_pos_y)
        return self

    ######################################
    ###### Array Wegiht Calculation ######
    ######################################
    def man_weight(self, amplitude, phase):
        self.aw_amp = amplitude.reshape(self.element_num_y,self.element_num_x)
        self.aw_phase = phase.reshape(self.element_num_y,self.element_num_x)
        self.array_weight = self.aw_amp * np.exp(1j*self.aw_phase*np.pi/180)
        self.excitation_manner = 'manual'
        return self

    def oam_weight(self, l):
        if l==0:
            self.aw_phase = np.zeros(self.element_num)
        else:
            self.aw_phase = np.arange(0,360*l,360*l/self.element_num)

        self.aw_power = np.ones(self.element_num)
        self.array_weight = np.sqrt(self.aw_power) * np.exp(1j*self.aw_phase*np.pi/180)

        return self

    def turn_off_element(self, x_index, y_index):
        aw_temp = self.array_weight
        aw_temp[y_index,x_index] = 0
        self.array_weight = aw_temp/np.sum(abs(aw_temp)**2)
        return self


    def beam_steering(self,theta_x=0, theta_y=0):
        phase_delay_x = np.exp(-1j*self.wavenumber*self.antenna_pos_x*np.sin(np.deg2rad(theta_x))).reshape(1,-1)
        phase_delay_y = np.exp(-1j*self.wavenumber*self.antenna_pos_y*np.sin(np.deg2rad(theta_y))).reshape(-1,1)
        aw_temp = self.array_weight * phase_delay_x * phase_delay_y
        self.array_weight = aw_temp
        self.excitation_manner = 'steered-beam'
        return self

    def beam_focus(self,focus_range):
        focus_range_ = focus_range/1000
        path_x = np.sqrt(self.antenna_pos_x_mesh**2 + focus_range_**2)
        path_xdif = path_x - focus_range_
        phase_xdif = self.wavenumber * path_xdif
        path_y = np.sqrt(self.antenna_pos_y_mesh**2 + focus_range_**2)
        path_ydif = path_y - focus_range_
        phase_ydif = self.wavenumber * path_ydif
        self.array_weight = self.array_weight * np.exp(1j*phase_xdif) * np.exp(1j*phase_ydif)
        if self.excitation_manner=='chebyshev':
            self.excitation_manner = 'chebyshev-focus/null'+str(null_angle)
        elif self.excitation_manner=='taylor':
            self.excitation_manner = 'taylor-focus'
        else:
             self.excitation_manner = 'focused-beam'
        return self

    def multi_beam_synthesis(self,frange,sangle,weight,name, ID=0):
        w = np.sqrt(weight / np.sum(weight))
        self.dir_flag = True
        aw_ini = self.array_weight
        aw_temp = np.tile(aw_ini, (frange.size,1,1))
        for i in range(sangle.size):
            phase_delay_x = np.exp(-1j*self.wavenumber*self.antenna_pos_x*np.sin(np.deg2rad(0))).reshape(1,-1)
            phase_delay_y = np.exp(-1j*self.wavenumber*self.antenna_pos_y*np.sin(np.deg2rad(sangle[i]))).reshape(-1,1)
            phase_delay = phase_delay_x * phase_delay_y
            path = np.sqrt(self.antenna_pos_x**2 + self.antenna_pos_y**2 + frange[i]**2)
            path_difference = path - frange[i]
            phase_difference = np.exp(1j * self.wavenumber * path_difference)
            aw_temp[i,:,:] = w[i] * phase_delay * phase_difference
        self.array_weight = np.sum(aw_temp,axis=0) / np.sqrt(aw_ini.size)
        self.excitation_manner = 'multi-beam-synthesis/' + name + '/ID-' + str(ID)
        return self

    def woodward_weight(self,sample_x,sample_y): # Unit of sample values is 'deg'
        sample_x_mesh, sample_y_mesh = np.meshgrid(np.deg2rad(sample_x),np.deg2rad(sample_y))
        sample_x_1D = sample_x_mesh.flatten()
        sample_y_1D = sample_y_mesh.flatten()
        aw_before = self.array_weight
        aw_temp = np.tile(aw_before, (sample_x_1D.size,1,1))
        for i in range(aw_temp.shape[0]):
            phase_delay_x = np.exp(-1j*self.wavenumber*self.antenna_pos_x*np.sin(sample_x_1D[i])).reshape(1,-1)
            phase_delay_y = np.exp(-1j*self.wavenumber*self.antenna_pos_y*np.sin(sample_y_1D[i])).reshape(-1,1)
            aw_temp[i,:,:] = aw_before * phase_delay_x * phase_delay_y

        aw_after = np.sum(aw_temp,axis=0)
        self.array_weight = aw_after / np.sqrt(np.sum(abs(aw_after)**2))
        return self

    def gauss_window(self, taper_db_x=False, taper_db_y=False):
        aw_before = self.array_weight
        xwindow = np.ones(aw_before.shape)
        ywindow = np.ones(aw_before.shape)
        if taper_db_x:
            Nx = self.element_num_x
            nx = np.arange(Nx)+1
            xwindow = np.exp(-np.log(10**(taper_db_x/20))*(2*(nx-1)/(Nx-1)-1)**2)
        if taper_db_y:
            Ny = self.element_num_y
            ny = np.arange(Ny)+1
            ywindow_temp = np.exp(-np.log(10**(taper_db_y/20))*(2*(ny-1)/(Ny-1)-1)**2).reshape(self.element_num_y,1)
            ywindow = np.tile(ywindow_temp, (1,self.element_num_x))
        aw_after = aw_before * xwindow * ywindow
        self.array_weight = aw_after / np.sqrt(np.sum(abs(aw_after)**2))
        return self

    def chebyshev_window(self, null_angle_x=False, null_angle_y=False):
        if (not null_angle_x) & (not null_angle_y):
            return self
        else:
            aw_before = self.array_weight
            xwindow = np.ones(self.antenna_pos_x.shape)
            ywindow = np.ones(self.antenna_pos_y.shape)
            if null_angle_x:
                null_rad_x = np.deg2rad(null_angle_x)
                Mx = self.element_num_x - 1
                x0 = np.cos(np.pi*self.element_int_wl_x*np.sin(null_rad_x))
                zx0 = 1./x0 * np.cos(np.pi/(2*Mx))
                xwindow = self.get_chebyshev(zx0,Mx).reshape(1,self.element_num_x)
            if null_angle_y:
                null_rad_y = np.deg2rad(null_angle_y)
                My = self.element_num_y - 1
                y0 = np.cos(np.pi*self.element_int_wl_y*np.sin(null_rad_y))
                zy0 = 1./y0 * np.cos(np.pi/(2*My))
                ywindow = self.get_chebyshev(zy0,My).reshape(self.element_num_y,1)
            aw_after = aw_before * (ywindow * xwindow)
            self.array_weight = aw_after / np.sqrt(np.sum(abs(aw_after)**2))
            return self

    def cosine_window(self,x_on=True,y_on=True):
        AW_ini = self.array_weight
        xwindow = np.ones(AW_ini.shape)
        ywindow = np.ones(AW_ini.shape)
        if x_on:
            xwindow = np.cos(np.pi*self.antenna_pos_x_mesh/2/self.antenna_pos_x_mesh.max())
        if y_on:
            ywindow = np.cos(np.pi*self.antenna_pos_y_mesh/2/self.antenna_pos_y_mesh.max())
        AW_temp = AW_ini * xwindow * ywindow
        self.array_weight = AW_temp / np.sqrt((AW_temp**2).sum())
        self.excitation_manner = 'cosine_window'
        return self

    def cosine_rolloff(self,alpha=1,x_on=True,y_on=True):
        AW_ini = self.array_weight
        xwindow = np.ones(AW_ini.shape)
        ywindow = np.ones(AW_ini.shape)
        if x_on:
            index_x1 = (abs(self.antenna_pos_x_mesh)<=(1+alpha)*self.antenna_pos_x_mesh.max()/2) \
                        & (abs(self.antenna_pos_x_mesh)>=(1-alpha)*self.antenna_pos_x_mesh.max()/2)
            index_x2 = abs(self.antenna_pos_x_mesh) >= (1+alpha)*self.antenna_pos_x_mesh.max()/2
            xwindow[index_x1] = np.cos(np.pi/2/alpha/self.antenna_pos_x_mesh.max()\
                        *(abs(self.antenna_pos_x_mesh[index_x1])-(1-alpha)*self.antenna_pos_x_mesh.max()/2))**2
            xwindow[index_x2] = 0
        if y_on:
            index_y1 = (abs(self.antenna_pos_y_mesh)<=(1+alpha)*self.antenna_pos_y_mesh.max()/2)\
                        & (abs(self.antenna_pos_y_mesh)>=(1-alpha)*self.antenna_pos_y_mesh.max()/2)
            index_y2 = abs(self.antenna_pos_y_mesh) >= (1+alpha)*self.antenna_pos_y_mesh.max()/2
            ywindow[index_y1] = np.cos(np.pi/2/alpha/self.antenna_pos_y_mesh.max()\
                        *(abs(self.antenna_pos_y_mesh[index_y1])-(1-alpha)*self.antenna_pos_y_mesh.max()/2))**2
            ywindow[index_y2] = 0
        AW_temp = AW_ini * xwindow * ywindow
        self.array_weight = AW_temp / np.sqrt((AW_temp**2).sum())
        self.excitation_manner = 'cosine_rolloff/alpha'+str(alpha)
        return self

    def happ_genzel(self,x_on=True,y_on=True):
        AW_ini = self.array_weight
        xwindow = np.ones(AW_ini.shape)
        ywindow = np.ones(AW_ini.shape)
        if x_on:
            xwindow = 0.54 + 0.46*np.cos(np.pi*self.antenna_pos_x_mesh/self.antenna_pos_x_mesh.max())
        if y_on:
            ywindow = 0.54 + 0.46*np.cos(np.pi*self.antenna_pos_y_mesh/self.antenna_pos_y_mesh.max())
        AW_temp = AW_ini * xwindow * ywindow
        self.array_weight = AW_temp / np.sqrt((AW_temp**2).sum())
        self.excitation_manner = 'happ_genzel'
        return self

    def array_weight_profile(self):
        ant_xpos = np.round(self.antenna_pos_x_mesh.flatten().reshape(self.element_num_x*self.element_num_y,1)*1000,1)
        ant_ypos = np.round(self.antenna_pos_y_mesh.flatten().reshape(self.element_num_x*self.element_num_y,1)*1000,1)
        amplitude_mat = abs(self.array_weight)
        amplitude = abs(self.array_weight.flatten()).reshape(self.element_num_x*self.element_num_y,1)
        phase_mat = np.round(np.rad2deg(np.angle(self.array_weight)),2)
        phase = np.round(np.rad2deg(np.angle(self.array_weight.flatten())).reshape(self.element_num_x*self.element_num_y,1),1)
        weight = np.hstack([ant_xpos,ant_ypos,amplitude,phase])
        self.aw_profile = pd.DataFrame(weight,columns=['X [mm]','Y [mm]','Amplitude [lin]','Phase [deg]'])
        return self

    def array_weight_cst(self,zoffset=0,split_num=2,sequential_block=False,post_fix=False):
        today_ = datetime.datetime.today()
        today = today_.strftime('%y%m%d')
        x = np.round(self.antenna_pos_x.reshape(-1,1),4)
        y = np.round(self.antenna_pos_y.reshape(-1,1),4)
        z = np.round(np.ones((self.element_num_x*self.element_num_y,1))*zoffset/1000,4)
        # cord = np.hstack([x,y,z])
        # df_cord = pd.DataFrame(cord,columns=['X','Y','Z'])

        amp = abs(self.array_weight).reshape(-1,1)
        if sequential_block:
            block_x, block_y = self.element_num_x // split_num, self.element_num_y // split_num
            phasemat = np.round(np.rad2deg(np.angle(self.array_weight)),1)
            print(block_x, block_y)
            for mm in range(split_num):
                for nn in range(split_num):
                    phasemat[mm * block_y:(mm * block_y + block_y // 2), nn * block_x:(nn * block_x + block_x // 2)] = \
                        phasemat[mm * block_y:(mm * block_y + block_y // 2),
                               nn * block_x:(nn * block_x + block_x // 2)]
                    phasemat[mm * block_y:(mm * block_y + block_y // 2), (nn * block_x + block_x // 2):(nn + 1) * block_x] = \
                        phasemat[mm * block_y:(mm * block_y + block_y // 2),
                               (nn * block_x + block_x // 2):(nn + 1) * block_x] - 90
                    phasemat[(mm * block_y + block_y // 2):(mm + 1) * block_y, (nn * block_x + block_x // 2):(nn + 1) * block_x] = \
                        phasemat[(mm * block_y + block_y // 2):(mm + 1) * block_y,
                               (nn * block_x + block_x // 2):(nn + 1) * block_x] - 180
                    phasemat[(mm * block_y + block_y // 2):(mm + 1) * block_y, nn * block_x:(nn * block_x + block_x // 2)] = \
                        phasemat[(mm * block_y + block_y // 2):(mm + 1) * block_y,
                               nn * block_x:(nn * block_x + block_x // 2)] - 270
            print(phasemat.shape)
            phase = phasemat.reshape(-1,1)
            print(phase)
        else:
            phase = np.round(np.rad2deg(np.angle(self.array_weight)),1).reshape(-1,1)
        # weight = np.hstack([amp,phase])
        # df_weight = pd.DataFrame(weight,columns=['Magnitude','Phase'])
        if sequential_block:
            block_x, block_y = self.element_num_x//split_num, self.element_num_y//split_num
            phimat = np.zeros((self.element_num_y,self.element_num_x))
            phimat_block = np.zeros((block_y, block_x))
            phi1 = phimat_block[:block_y//2, :block_x//2]
            phi2 = phimat_block[:block_y//2, block_x//2:]+90
            phi3 = phimat_block[block_y//2:, block_x//2:]+180
            phi4 = phimat_block[block_y//2:, :block_x//2]+270
            block_phi = np.vstack([np.hstack([phi1,phi2]),np.hstack([phi4,phi3])])
            # print(block_phi)
            for m in range(split_num):
                for n in range(split_num):
                    phimat[block_y*m:block_y*(m+1), block_x*n:block_x*(n+1)] = block_phi
            phi = phimat.reshape(-1, 1)
        else:
            phi = np.zeros(self.element_num_x*self.element_num_y).reshape(-1,1)
        theta = np.zeros(self.element_num_x*self.element_num_y).reshape(-1,1)
        gamma = np.zeros(self.element_num_x*self.element_num_y).reshape(-1,1)
        # rot = np.hstack([phi,theta,gamma])
        # df_rot = pd.DataFrame(rot,columns=['Phi','Theta','Gamma'])
        df_num = pd.DataFrame(np.arange(self.element_num_x*self.element_num_y,dtype=np.int64).reshape(-1,1))
        df_full = pd.DataFrame(np.hstack([x,y,z,amp,phase,phi,theta,gamma]))
        df = pd.concat([df_num,df_full],axis=1)
        os.makedirs('weight/{0}/{1}'.format(self.excitation_manner,today),exist_ok=True)
        if post_fix:
            filename = 'cst_import_{}'.format(post_fix)
        else:
            filename = 'cst_import'
        df.T.set_index(pd.MultiIndex(levels=[["# Created by",""],["# Element","X","Y","Z","Magnitude","Phase","Phi","Theta","Gamma"]],\
                codes=[[0,1,1,1,1,1,1,1,1],[0,1,2,3,4,5,6,7,8]])).T.\
                to_csv('weight/{}/{}/{}.tsv'.format(self.excitation_manner,today,filename),sep='\t',index=False)

        return self

    def array_weight_plot(self,axis='x',row=0,fs=20):
        amp_ = abs(self.array_weight)
        phase_ = np.rad2deg(np.angle(self.array_weight))
        if axis=='x':
            pos = self.antenna_pos_x[row,:]
            amp = amp_[row,:]
            phase = phase_[row,:]
        elif axis=='y':
            pos = self.antenna_pos_y[:,row]
            amp = amp_[:,row]
            phase = phase_[:,row]
        else:
            print('axis error')
            sys.exit()
        cmap = plt.get_cmap("tab10")
        amp_fig, amp_ax = plt.subplots(figsize=(12,8))
        phase_fig, phase_ax = plt.subplots(figsize=(12,8))
        amp_ax.plot(pos,amp,'o',color=cmap(1),ms=12,linestyle='None')
        phase_ax.plot(pos,phase,'o',color=cmap(0),ms=12,linestyle='None')
        amp_ax.set_xlabel('Position [mm]',fontsize=fs)
        amp_ax.set_ylabel('Amplitude',fontsize=fs)
        amp_ax.set_ylim([-0.05,1.05])
        amp_ax.tick_params(axis='x',labelsize=fs)
        amp_ax.tick_params(axis='y',labelsize=fs)
        phase_ax.set_xlabel('Position [mm]',fontsize=fs)
        phase_ax.set_ylabel('Phase [deg]',fontsize=fs)
        phase_ax.set_ylim([-180,180])
        phase_ax.tick_params(axis='x',labelsize=fs)
        phase_ax.tick_params(axis='y',labelsize=fs)
        self.aw_amp_figure = amp_fig
        self.aw_phase_figure = phase_fig
        self.aw_which_row = axis+str(row)
        return self

    def weight_check(self):
        print((abs(self.array_weight)**2).sum())
        return self

    def get_chebyshev(self,z0,M):
        if np.mod(M,2): #even
            N = int((M+1)/2)
            I = np.zeros(N)
            for i in range(N):
                q = N - i
                T1 = self.get_A(2*q-1,M)*z0**(2*q-1)
                T2 = 0
                for k in np.arange(q+1,N+1):
                    T2 = T2 + I[k-1]*self.get_A(2*q-1,2*k-1)
                I[q-1] = 1/self.get_A(2*q-1,2*q-1) * (T1 - T2)
            I2= np.block([np.flip(I,axis=0),I])
            AW = abs(I2) * np.exp(1j*np.angle(I2))
            return AW / np.sqrt(np.sum(abs(I2)**2))

        else: #odd
            N = int(M/2)
            I = np.zeros(N+1)
            for i in np.arange(1,N+1):
                q = N+1 - i
                T1 = self.get_A(2*q,M)*z0**(2*q)
                T2 = 0
                for k in np.arange(q+1,N+1):
                    T2 = T2 + I[k-1]*self.get_A(2*q,2*k)
                I[q-1] = 1/self.get_A(2*q,2*q) * (T1 - T2)
            I2 = np.block([np.flip(I[1:],axis=0),I])
            AW = abs(I2) * np.exp(1j*np.angle(I2))
            return AW / np.sum(abs(I2)) * np.size(AW)

    def get_A(self,a,b):
        if (np.mod(a,2)==1) & (np.mod(b,2)==1):
            m = (a-1)/2
            n = (b-1)/2
            A = 0
            for p in np.arange(n-m,n+1):
                A = A + (C(p,p-n+m) * C(2*n+1,2*p))
            return A*(-1)**(n-m)
        elif (np.mod(a,2)==0) & (np.mod(b,2)==0):
            m = a/2
            n = b/2
            A = 0
            for p in np.arange(n-m,n+1):
                A = A + (C(p,p-n+m) * C(2*n,2*p))
            return A*(-1)**(n-m)
        else:
            print('Error')

    ############################
    ###### Element Factor ######
    ############################
    def set_element(self, filename='isotropic',software='cst'):
        if filename=='isotropic':
            # self.EP = 1
            self.element_factor_theta = np.ones((181,361))/np.sqrt(2)
            self.element_factor_phi = np.ones((181,361))/np.sqrt(2)
            self.element_factor_abs = np.ones((181,361))
        elif os.path.exists('data/farfield/{}_theta_abs.csv'.format(filename)):
            EFtheta_mag = pd.read_csv('data/farfield/{}_theta_abs.csv'.format(filename), header=None).values
            EFtheta_angle = pd.read_csv('data/farfield/{}_theta_angle.csv'.format(filename), header=None).values
            EFphi_mag = pd.read_csv('data/farfield/{}_phi_abs.csv'.format(filename), header=None).values
            EFphi_angle = pd.read_csv('data/farfield/{}_phi_angle.csv'.format(filename), header=None).values
            EFabs = pd.read_csv('data/farfield/{}_abs.csv'.format(filename), header=None).values
            self.element_factor_theta = EFtheta_mag * np.exp(1j*EFtheta_angle)
            self.element_factor_phi = EFphi_mag * np.exp(1j*EFphi_angle)
            self.element_factor_abs = np.sqrt(EFtheta_mag**2 + EFphi_mag**2)
        else:
            if software=='cst':
                df = pd.read_table('data/farfield/{}.txt'.format(filename), header=None, skiprows=2, delimiter='\s+')
                data = df.values

                data0 = data[:,[0,1,2]] # Gain (Linear, unit:W)
                data1 = data[:,[0,1,3,4]]  #theta
                data2 = data[:,[0,1,5,6]]  #phi

                theta_num = 181
                phi_num = 361
                Etheta = np.zeros((theta_num,phi_num),dtype='complex128')
                Ephi = np.zeros((theta_num,phi_num),dtype='complex128')
                Eabs = np.zeros((theta_num,phi_num),dtype='float64')
                angle = np.zeros((theta_num,phi_num))
                for i in range(theta_num):
                    for j in range(phi_num):
                        theta = i
                        phi = j
                        if (theta == 180) and ((phi<=90) or (phi>270)):
                            theta = -180

                        if (phi > 90) and ( phi <= 270) :
                            theta = -theta
                            phi = phi - 180

                        if phi > 270 :
                            phi = phi - 360

                        index1 = np.where(data1[:,0]==theta)
                        index2 = np.where(data1[:,1]==phi)
                        index3 = np.intersect1d(index1,index2)

                        Etheta[i,j] = np.sqrt(data1[index3,2]) * np.exp(1j*data1[index3,3]*np.pi/180) # Unit:V
                        Ephi[i,j] = np.sqrt(data2[index3,2]) * np.exp(1j*data2[index3,3]*np.pi/180) # Unit:V
                        angle[i,j] = data2[index3,3]
                        Eabs[i,j] = np.sqrt(data0[index3,2])

                Etheta_mag = abs(Etheta)
                Etheta_angle = np.angle(Etheta)
                Ephi_mag = abs(Ephi)
                Ephi_angle = np.angle(Ephi)


            pd.DataFrame(Etheta_mag).to_csv('data/farfield/' + filename + '_theta_abs.csv',header=None,index=None)
            pd.DataFrame(Etheta_angle).to_csv('data/farfield/' + filename + '_theta_angle.csv',header=None,index=None)
            pd.DataFrame(Ephi_mag).to_csv('data/farfield/' + filename + '_phi_abs.csv',header=None,index=None)
            pd.DataFrame(Ephi_angle).to_csv('data/farfield/' + filename + '_phi_angle.csv',header=None,index=None)
            pd.DataFrame(Eabs).to_csv('data/farfield/' + filename + '_abs.csv',header=None,index=None)
            self.element_factor_theta = Etheta
            self.element_factor_phi = Ephi
            self.element_factor_abs = Eabs

        return self

    ##################################
    ###### Farfield Calculation ######
    ##################################
    def plot_farfield(self, phi = 0, distance=0, angle = (-90,90), gain = False, angle_ticks = False, gain_ticks = False, filename = "ff_gain", ext=".png"):
        # Calculation angle definition
        theta_step_num = self.element_factor_theta.shape[0]
        phi_step_num = self.element_factor_theta.shape[1]
        theta_step = np.pi/(theta_step_num-1)
        phi_step = 2*np.pi/(phi_step_num-1)
        theta_range = np.arange(0, np.pi+theta_step, theta_step)
        phi_range = np.arange(0, 2*np.pi+phi_step, phi_step)
        self.phi_mesh, self.theta_mesh = np.meshgrid(phi_range, theta_range)
        # Calculation distance definition
        if distance==0:
            antenna_total_xwidth = self.element_num_x*self.element_int_x/1000
            antenna_total_ywidth = self.element_num_y*self.element_int_y/1000
            antenna_total_width = max([antenna_total_xwidth,antenna_total_ywidth])
            farfield_range = 4 * antenna_total_width**2 /self.wavelength
        else:
            farfield_range = distance
        # Array factor calculation
        AW = self.array_weight.flatten()
        ant_xpos = self.antenna_pos_x_mesh.flatten()
        ant_ypos = self.antenna_pos_y_mesh.flatten()
        self.array_factor = np.zeros(self.theta_mesh.shape)
        for i in range(self.array_weight.size):
            self.array_factor = self.array_factor + AW[i] * np.exp(1j*self.wavenumber*(ant_xpos[i]*np.sin(self.theta_mesh)*np.cos(self.phi_mesh)\
                        + ant_ypos[i]*np.sin(self.theta_mesh)*np.sin(self.phi_mesh)))
        # calibration factor calculation
        gridsize = (farfield_range * theta_step) * (farfield_range * np.sin(self.theta_mesh) * phi_step)
        AF_mag = abs(self.array_factor)
        AW_sum = np.sum(abs(self.array_weight)**2)
        EFabs2 = abs(self.element_factor_abs)**2
        N = AW_sum * np.sum(EFabs2 * gridsize)
        D = np.sum(EFabs2 * AF_mag**2 * gridsize)
        self.calibration_factor = np.sqrt(N/D)
        # Multiplying the three facotrs
        self.farfield_theta = self.calibration_factor * self.array_factor * self.element_factor_theta
        self.farfield_phi = self.calibration_factor * self.array_factor * self.element_factor_phi
        self.farfield_abs = abs(self.calibration_factor * self.array_factor * self.element_factor_abs)


        X1 = self.theta_mesh[:abs(angle[0])+1,int(phi)]*180/np.pi
        X2 = self.theta_mesh[:abs(angle[1])+1,int(phi)+int((self.theta_mesh.shape[1]-1)/2)]*180/np.pi
        x = np.block([-np.flip(X2[1:], axis=0), X1])

        Y1 = self.farfield_abs[:abs(angle[0])+1,int(phi)]
        Y2 = self.farfield_abs[:abs(angle[1])+1,int(phi)+int((self.theta_mesh.shape[1]-1)/2)]
        Y = np.block([np.flip(Y2[1:], axis=0), Y1])
        y = 20*np.log10(Y)
        self.farfield_plot_angle = np.round(x)
        self.farfield_plot_gain = np.round(y,2)
        self.maximum_gain = np.round(self.farfield_plot_gain.max(),1)
        self.maximum_gain_angle = self.farfield_plot_angle[self.farfield_plot_gain.argmax()]
        print('Directive Gain: {} dBi\nMain Lobe Angle: {} degree'.format(self.maximum_gain,self.maximum_gain_angle))
        plt.rcParams["font.size"] = 20
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        ax.plot(x,y)
        if angle:
            xmin = angle[0]
            xmax = angle[1]
        else:
            xmin = -180
            xmax = 180
        ax.set_xlim([xmin,xmax])
        if gain:
            ymin = gain[0]
            ymax = gain[1]
            ax.set_ylim([ymin,ymax])
        if angle_ticks:
            ax.set_xticks(angle_ticks)
        if gain_ticks:
            ax.set_yticks(gain_ticks)
        ax.set_xlabel("Angle [deg]",fontsize=20)
        ax.set_ylabel("Gain [dB]",fontsize=20)
        ax.grid(which="both",c="gainsboro", zorder=9)
        self.farfield_fig = fig
        os.makedirs("synthesis-result/"+today+"/farfield",exist_ok=True)
        self.farfield_fig.savefig("synthesis-result/"+today+"/farfield/"+filename+ext)
        pd.DataFrame(np.stack([self.farfield_plot_angle,self.farfield_plot_gain]).T,columns=["Angle (deg)","Gain (dBi)"]).to_csv("synthesis-result/"+today+"/farfield/"+filename+".csv",index=False)
        return self


    def first_sidelobe(self): # first side lobe angle and level
        angle_front = self.farfield_plot_angle[(self.farfield_plot_angle>=-90)&(self.farfield_plot_angle<=90)]
        gain_front = self.farfield_plot_gain[(self.farfield_plot_angle>=-90)&(self.farfield_plot_angle<=90)]
        ml_angle = angle_front[gain_front.argmax()]
        gain_max = gain_front.max()
        sl_angle = self.farfield_plot_angle[signal.argrelmax(self.farfield_plot_gain)]
        sll = self.farfield_plot_gain[signal.argrelmax(self.farfield_plot_gain)] - gain_max

        self.first_sidelobe_angle = np.round(sl_angle[np.where(sl_angle==ml_angle)[0] + 1],1)
        self.first_sidelobe_level = np.round(sll[np.where(sl_angle==ml_angle)[0] + 1],1)
        print('First Side Lobe Lebel: '+str(self.first_sidelobe_level[0])+' dB\nFirst Side Lobe Angle: ' + str(self.first_sidelobe_angle[0])+' degree')
        return self

    def sidelobe(self): # first side lobe angle and level
        angle_front = self.farfield_plot_angle[(self.farfield_plot_angle>=-90)&(self.farfield_plot_angle<=90)]
        gain_front = self.farfield_plot_gain[(self.farfield_plot_angle>=-90)&(self.farfield_plot_angle<=90)]
        ml_angle = angle_front[gain_front.argmax()]
        gain_max = gain_front.max()
        sidelobe_angle = self.farfield_plot_angle[signal.argrelmax(self.farfield_plot_gain)]
        sidelobe_level = self.farfield_plot_gain[signal.argrelmax(self.farfield_plot_gain)] - gain_max
        sidelobe = np.vstack([sidelobe_angle,sidelobe_level])
        for i, (sla, sll) in enumerate(zip(sidelobe)):
            print("{0}.   Side Lobe Angle: {1} deg   Side Lobe Level: {2} dB\n".format(i,sla,sll))
        return sidelobe

    def null(self): # null angle
        angle_front = self.farfield_plot_angle[(self.farfield_plot_angle>=-90)&(self.farfield_plot_angle<=90)]
        gain_front = self.farfield_plot_gain[(self.farfield_plot_angle>=-90)&(self.farfield_plot_angle<=90)]
        ml_angle = angle_front[gain_front.argmax()]
        null_angle = angle_front[signal.argrelmin(gain_front)]
        for i, nla in enumerate(null_angle):
            print("{0}.   Null Angle: {1} deg\n".format(i,null_angle))
        return self


    def half_power_beamwidth(self):
        angle_front = self.farfield_plot_angle[(self.farfield_plot_angle>=-90)&(self.farfield_plot_angle<=90)]
        gain_front = self.farfield_plot_gain[(self.farfield_plot_angle>=-90)&(self.farfield_plot_angle<=90)]
        gain_max = gain_front.max()
        gain0dB = gain_front - gain_max
        anglehalf = angle_front[np.ceil(gain0dB) >= -3.] # pick up components with more than -3dB gain rounded up
        gainhalf = 10**(gain0dB[np.ceil(gain0dB) >= -3.]/10) # Linear scale
        anglf = anglehalf[0]
        anglc = anglehalf[1]
        angrf = anglehalf[-2]
        angrc = anglehalf[-1]
        gainlf = gainhalf[0]
        gainlc = gainhalf[1]
        gainrf = gainhalf[-2]
        gainrc = gainhalf[-1]
        hpbw_left = round(anglf * (gainlc-0.5)/(gainlc-gainlf) + anglc * (0.5-gainlf)/(gainlc-gainlf),1)
        hpbw_right = round(angrf * (0.5-gainrc)/(gainrf-gainrc) + angrc * (gainrf-0.5)/(gainrf-gainrc),1)
        self.beamwidth = hpbw_right-hpbw_left
        print('Half Power Beam Width: '+str(self.beamwidth)+' degree')
        return self

    def farfield_profile(self):
        gain = self.maximum_gain
        gain_angle = self.maximum_gain_angle
        bw = self.beamwidth
        fsll = self.first_sidelobe_level
        fsla = self.first_sidelobe_angle
        profile = np.array([gain,gain_angle,bw,fsll[0],fsla[0]]).reshape(5,1)
        self.farfield_dataframe = pd.DataFrame(profile,index=['Gain(dBi)','Angle(deg)','HPBW(deg)','SLL(dB)','SLA(dB)'])
        return self


    def get_index(self):
        index = np.zeros(y.size)
        for i in range(y.size):
            index[i] = np.where(x==y[i])[0]
        return index.astype(np.int64)

    def set_input_power(self,Pin):
        self.input_power = Pin
        return self
    ###################################
    ###### Nearfield Calculation ######
    ###################################
    def plot_nearfield(self, xlim, ylim, z, receiving_area = False, rx_offset = False, xticks=False, yticks=False, clevel=False, cticks=False, rline_width=5, divnum_x=1000, divnum_y=1000):
        xrange = np.linspace(xlim[0],xlim[1],num=divnum_x+1,endpoint=True)/1000 #[m]
        yrange = np.linspace(ylim[0],ylim[1],num=divnum_y+1,endpoint=True)/1000 #[m]
        xrange_mesh, yrange_mesh = np.meshgrid(np.round(xrange,decimals=4), np.round(yrange,decimals=4))
        z_ = z/1000
        self.nearfield_x_mesh = xrange_mesh
        self.nearfield_y_mesh = yrange_mesh
        theta_center = np.arctan(np.sqrt((xrange_mesh**2+yrange_mesh**2)/z_))
        array_weight = self.array_weight.flatten()
        ant_xpos = self.antenna_pos_x_mesh.flatten()
        ant_ypos = self.antenna_pos_y_mesh.flatten()
        center_to_target = np.sqrt(xrange_mesh**2 + yrange_mesh**2 + z_**2)
        element_num = array_weight.size

        # E_theta_sum = np.zeros((element_num,xrange_mesh.shape[0],xrange_mesh.shape[1]),dtype='complex128')
        # E_phi_sum = np.zeros((element_num,xrange_mesh.shape[0],xrange_mesh.shape[1]),dtype='complex128')
        electric_field = np.zeros((element_num,xrange_mesh.shape[0],xrange_mesh.shape[1]),dtype='complex128')

        for i in range(element_num):
            element_xrange_mesh = xrange_mesh-ant_xpos[i]
            element_yrange_mesh = yrange_mesh-ant_ypos[i]
            element_to_target = np.sqrt(element_xrange_mesh**2 + element_yrange_mesh**2 + z_**2)
            theta = np.arctan(np.sqrt(element_xrange_mesh**2+element_yrange_mesh**2)/z_)
            phi_temp = np.arctan2(element_yrange_mesh, element_xrange_mesh)
            phi = np.where(phi_temp>=0,phi_temp,2*np.pi+phi_temp)
            path_difference = center_to_target - element_to_target
            array_factor = array_weight[i] * np.exp(1j*self.wavenumber*path_difference)/element_to_target

            theta_deg = np.round(np.rad2deg(theta),decimals=1)
            thetaf = np.floor(theta_deg).astype(np.int64)
            thetac = np.ceil(theta_deg).astype(np.int64)
            alphat = theta_deg - thetaf

            phi_deg = np.round(np.rad2deg(phi),decimals=1)
            phif = np.floor(phi_deg).astype(np.int64)
            phic = np.ceil(phi_deg).astype(np.int64)
            alphap = phi_deg - phif

            # EFtheta = self.element_factor_theta[thetaf,phif] * (1-alphat) * (1-alphap)+ self.element_factor_theta[thetac,phif] * alphat * (1-alphap) \
            #         + self.element_factor_theta[thetaf,phic] * (1-alphat) * alphap + self.element_factor_theta[thetac,phic] * alphat * alphap
            # EFphi = self.element_factor_phi[thetaf,phif] * (1-alphat) * (1-alphap)+ self.element_factor_phi[thetac,phif] * alphat * (1-alphap) \
            #         + self.element_factor_phi[thetaf,phic] * (1-alphat) * alphap + self.element_factor_phi[thetac,phic] * alphat * alphap

            element_factor_abs = self.element_factor_abs[thetaf,phif] * (1-alphat) * (1-alphap)+ self.element_factor_abs[thetac,phif] * alphat * (1-alphap) \
                    + self.element_factor_abs[thetaf,phic] * (1-alphat) * alphap + self.element_factor_abs[thetac,phic] * alphat * alphap


            # E_theta_sum[i,:,:] = EFtheta*AF # [V/m]
            # E_phi_sum[i,:,:] = EFphi*AF # [V/m]
            electric_field[i,:,:] = array_factor * element_factor_abs # [V/m]
        self.efield_phase = np.rad2deg(np.angle(np.sum(electric_field,axis=0)))
        self.power_density = abs(self.calibration_factor * np.sum(electric_field,axis=0))**2 * self.input_power/(4*np.pi) * np.cos(theta_center) * 0.1

        plt.rcParams["font.size"] = 26
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        if clevel:
            levels_ = np.linspace(start=0, stop=clevel, num=100, endpoint=True)
        else:
            levels_ = np.linspace(start=0, stop=round(self.power_density.max(),1), num=100, endpoint=True)
        ax.set_aspect('equal')
        xmin = xlim[0]
        xmax = xlim[1]
        ax.set_xlim([xmin,xmax])
        ymin = ylim[0]
        ymax = ylim[1]
        ax.set_ylim([ymin,ymax])
        ax.set_xlabel(r'$x\ {\rm [mm]}$')
        if xticks:
            ax.set_xticks(xticks)
        ax.set_ylabel(r'$y\ {\rm [mm]}$')
        if yticks:
            ax.set_yticks(yticks)
        if receiving_area:
            ax.hlines(y=[-receiving_area[1]/2, receiving_area[1]/2], xmin=-receiving_area[0]/2, xmax=receiving_area[0]/2, color='white',linewidth=rline_width)
            ax.vlines(x=[-receiving_area[0]/2, receiving_area[0]/2], ymin=-receiving_area[1]/2, ymax=receiving_area[1]/2, color='white',linewidth=rline_width)
        im = ax.contourf(self.nearfield_x_mesh*1000,self.nearfield_y_mesh*1000,self.power_density, levels=levels_, cmap='jet',extend='both') #x,y->[mm], pd->[mW/cm2]
        if cticks:
            cbar = fig.colorbar(im, ticks = cticks)
            cbar.set_label(r'$p_{\rm d}\ {\rm [mW/cm^2]}$')
        else:
            cticks=np.linspace(0, round(self.power_density.max(),0), num=5)
            cbar = fig.colorbar(im, ticks = cticks)
            cbar.set_label(r'$p_{\rm d}\ {\rm [mW/cm^2]}$')
        self.nf_2dcontour_ax = ax
        self.nf_2dcontour_fig = fig
        return self


    def calculate_received_power(self, xlim, ylim, z, divnum=500):
        xgrid = (xlim[1]-xlim[0])/1000/divnum
        ygrid = (ylim[1]-ylim[0])/1000/divnum
        xshift = xgrid/2
        yshift = ygrid/2
        xrange = np.linspace(xlim[0]/1000+xshift,xlim[1]/1000-xshift,num=divnum,endpoint=True)
        yrange = np.linspace(ylim[0]/1000+yshift,ylim[1]/1000-yshift,num=divnum,endpoint=True)
        xrange_mesh, yrange_mesh = np.meshgrid(xrange, yrange)
        z_ = z/1000

        theta_center = np.arctan(np.sqrt((xrange_mesh**2+yrange_mesh**2)/z_))
        array_weight = self.array_weight.flatten()
        ant_xpos = self.antenna_pos_x_mesh.flatten()
        ant_ypos = self.antenna_pos_y_mesh.flatten()
        center_to_target = np.sqrt(xrange_mesh**2 + yrange_mesh**2 + z_**2)
        element_num = array_weight.size
        electric_field = np.zeros((element_num,xrange_mesh.shape[0],xrange_mesh.shape[1]),dtype='complex128')
        for i in range(element_num):
            X = xrange_mesh-ant_xpos[i]
            Y = yrange_mesh-ant_ypos[i]
            element_to_target = np.sqrt(X**2 + Y**2 + z_**2)
            theta = np.arctan(np.sqrt(X**2+Y**2)/z_)
            phi_temp = np.arctan2(Y, X)
            phi = np.where(phi_temp>=0,phi_temp,2*np.pi+phi_temp)
            path_difference = center_to_target - element_to_target
            array_factor = array_weight[i] * np.exp(1j*self.wavenumber*path_difference)/element_to_target

            theta_deg = np.round(np.rad2deg(theta),decimals=1)
            thetaf = np.floor(theta_deg).astype(np.int64)
            thetac = np.ceil(theta_deg).astype(np.int64)
            alphat = theta_deg - thetaf

            phi_deg = np.round(np.rad2deg(phi),decimals=1)
            phif = np.floor(phi_deg).astype(np.int64)
            phic = np.ceil(phi_deg).astype(np.int64)
            alphap = phi_deg - phif

            element_factor_abs = self.element_factor_abs[thetaf,phif] * (1-alphat) * (1-alphap)+ self.element_factor_abs[thetac,phif] * alphat * (1-alphap) \
                    + self.element_factor_abs[thetaf,phic] * (1-alphat) * alphap + self.element_factor_abs[thetac,phic] * alphat * alphap
            electric_field[i,:,:] = array_factor * element_factor_abs # [V/m]

        self.power_density_on_rx = abs(self.calibration_factor * np.sum(electric_field,axis=0))**2 * self.input_power / (4*np.pi) * np.cos(theta_center) # W/m2
        self.total_received_power = np.sum(self.power_density_on_rx*xgrid*ygrid)
        pdmax = np.round(self.power_density_on_rx.max()/10,1) # mW/cm2
        ave = np.round(np.average(self.power_density_on_rx/10),1) # mW/cm2
        trp = np.round(self.total_received_power,1)
        eff = np.round(self.total_received_power/self.input_power *100,1)
        dev = np.round(np.std(self.power_density_on_rx)/10,1)
        print('Maximum Power: '+str(pdmax)+' mW/cm2\nAverage Power: '+str(ave)+' mW/cm2\nTotal Received Power: ' + str(trp)+' W\nBeam Efficiency: ' + str(eff)+' %\nStandard Deviation: '+str(dev)+' mW/cm2')
        self.efficiency_profile = pd.DataFrame([[pdmax],[ave],[eff],[dev]],index=['Max. Power Density [mW/cm2]','Ave. Power Density [mW/cm2]','Beam Efficiency [%]','Standard Deviation [mW/cm2]'])
        return self



#############################################################################################

    def pattern_3Dcontour(self, xlim, ylim, zlim, Pin, xstep=1, ystep=1 ,zstep=1):
        xlim = np.array(xlim)/1000
        ylim = np.array(ylim)/1000
        zlim = np.array(zlim)/1000
        xrange = np.arange(xlim[0],xlim[1]+xstep/1000,xstep/1000)
        yrange = np.arange(ylim[0],ylim[1]+ystep/1000,ystep/1000)
        zrange = np.arange(zlim[0],zlim[1]+zstep/1000,zstep/1000)
        xrange_mesh, yrange_mesh = np.meshgrid(np.round(xrange,decimals=3), np.round(yrange,decimals=3))
        self.nf_x_2dmesh = xrange_mesh*1000
        self.nf_y_2dmesh = yrange_mesh*1000
        self.nf_z_1drange = zrange*1000
        self.input_power = Pin
        Pd = np.zeros((zrange.size,yrange.size,xrange.size))

        for j in range(zrange.size):

            theta_center = np.arctan(np.sqrt((xrange_mesh**2+yrange_mesh**2)/zrange[j]))
            AW = self.AW.flatten()
            ant_xpos = self.antenna_pos_x_mesh.flatten()
            ant_ypos = self.antenna_pos_y_mesh.flatten()
            R_center = np.sqrt(xrange_mesh**2 + yrange_mesh**2 + zrange[j]**2)
            element_num = AW.size
            E_sum = np.zeros((element_num,yrange.size,xrange.size),dtype='complex128')
            for i in range(element_num):
                X = xrange_mesh-ant_xpos[i]
                Y = yrange_mesh-ant_ypos[i]
                R = np.sqrt(X**2 + Y**2 + zrange[j]**2)
                theta = np.arctan(np.sqrt(X**2+Y**2)/zrange[j])
                phi_temp = np.arctan2(Y, X)
                phi = np.where(phi_temp>=0,phi_temp,2*np.pi+phi_temp)
                AF = AW[i] * np.exp(1j*self.wavenumber*(R_center-R))
                theta_deg = np.round(np.rad2deg(theta),decimals=1)
                thetaf = np.floor(theta_deg).astype(np.int64)
                thetac = np.ceil(theta_deg).astype(np.int64)
                alphat = theta_deg - thetaf
                phi_deg = np.round(np.rad2deg(phi),decimals=1)
                phif = np.floor(phi_deg).astype(np.int64)
                phic = np.ceil(phi_deg).astype(np.int64)
                alphap = phi_deg - phif
                EFabs = self.element_factor_abs[thetaf,phif] * (1-alphat) * (1-alphap)+ self.element_factor_abs[thetac,phif] * alphat * (1-alphap) \
                        + self.element_factor_abs[thetaf,phic] * (1-alphat) * alphap + self.element_factor_abs[thetac,phic] * alphat * alphap
                E_sum[i,:,:] = AF * EFabs # [V/m]

            Pd[j,:,:] = abs(self.calibration_factor * np.sum(E_sum,axis=0))**2 * self.input_power / (4*np.pi*(R_center**2)) * np.cos(theta_center) * 0.1 # mW/cm2

        self.power_density_3d = Pd

        return self

    def plot_3dcut(self,xlim,ylim,z):
        plot3d = self.power_density_3d[self.nf_z_1drange==z,:,:]

        plt.rcParams["font.size"] = 26
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        levels_ = np.linspace(start=0, stop=round(plot3d.max(),1), num=100, endpoint=True)
        ax.set_aspect('equal')
        xmin = xlim[0]
        xmax = xlim[1]
        ax.set_xlim([xmin,xmax])
        ymin = ylim[0]
        ymax = ylim[1]
        ax.set_ylim([ymin,ymax])
        ax.set_xlabel(r'$x\ {\rm [mm]}$')
        # ax.set_xticks(xticks*1000)
        ax.set_ylabel(r'$y\ {\rm [mm]}$')
        # ax.set_yticks(yticks*1000)
        im = ax.contourf(self.nf_x_2dmesh,self.nf_y_2dmesh,plot3d[0], levels=levels_, cmap='jet',extend='both')
        cbar = fig.colorbar(im, ticks = np.linspace(0, round(plot3d.max(),1), num=5))
        cbar.set_label(r'$p_{\rm d}\ {\rm [mW/cm^2]}$')
        self.nf_2dcontour_ax = ax
        self.nf_2dcontour_fig = fig
        return self

    def export_3dcontour(self,name):
        xtemp = self.nf_x_2dmesh.flatten()
        x = np.tile(xtemp,self.nf_z_1drange.size)
        ytemp = self.nf_y_2dmesh.flatten()
        y = np.tile(ytemp,self.nf_z_1drange.size)
        ztemp = np.tile(self.nf_z_1drange.reshape(-1,1),(1,self.nf_x_2dmesh.size))
        z = ztemp.flatten()
        self.nf_x_3dmesh = x.reshape(self.nf_z_1drange.size,self.nf_x_2dmesh.shape[1],self.nf_x_2dmesh.shape[0])
        self.nf_y_3dmesh = y.reshape(self.nf_z_1drange.size,self.nf_x_2dmesh.shape[1],self.nf_x_2dmesh.shape[0])
        self.nf_z_3dmesh = z.reshape(self.nf_z_1drange.size,self.nf_x_2dmesh.shape[1],self.nf_x_2dmesh.shape[0])
        Pd = self.power_density_3d.flatten()
        df = pd.DataFrame(np.stack([x,y,z,Pd],axis=1),columns=['X','Y','Z','Power Density'])
        os.makedirs('results/3d',exist_ok=True)
        df.to_csv('results/3d/'+name+'.csv',index=False)
        return self

    def pattern_2Dcontour_test(self, xlim, ylim, z, divnum_x=1000, divnum_y=1000):
        xrange = np.linspace(xlim[0],xlim[1],num=divnum_x+1,endpoint=True)
        yrange = np.linspace(ylim[0],ylim[1],num=divnum_y+1,endpoint=True)
        xrange_mesh, yrange_mesh = np.meshgrid(np.round(xrange,decimals=4), np.round(yrange,decimals=4))
        self.nearfield_x_mesh = xrange_mesh
        self.nearfield_y_mesh = yrange_mesh

        theta_center = np.arctan(np.sqrt((xrange_mesh**2+yrange_mesh**2)/z))
        AW = self.AW.flatten()
        ant_xpos = self.antenna_pos_x_mesh.flatten()
        ant_ypos = self.antenna_pos_y_mesh.flatten()
        R_center = np.sqrt(xrange_mesh**2 + yrange_mesh**2 + z**2)
        element_num = AW.size
        E_theta_sum = np.zeros((element_num,xrange_mesh.shape[0],xrange_mesh.shape[1]),dtype='complex128')
        E_phi_sum = np.zeros((element_num,xrange_mesh.shape[0],xrange_mesh.shape[1]),dtype='complex128')
        Ex = np.zeros((element_num,xrange_mesh.shape[0],xrange_mesh.shape[1]),dtype='complex128')
        Ey = np.zeros((element_num,xrange_mesh.shape[0],xrange_mesh.shape[1]),dtype='complex128')
        Ez = np.zeros((element_num,xrange_mesh.shape[0],xrange_mesh.shape[1]),dtype='complex128')
        E_sum = np.zeros((element_num,xrange_mesh.shape[0],xrange_mesh.shape[1]),dtype='complex128')
        for i in range(element_num):
            X = xrange_mesh-ant_xpos[i]
            Y = yrange_mesh-ant_ypos[i]
            R = np.sqrt(X**2 + Y**2 + z**2)
            theta = np.arctan(np.sqrt(X**2+Y**2)/z)
            phi_temp = np.arctan2(Y, X)
            phi = np.where(phi_temp>=0,phi_temp,2*np.pi+phi_temp)
            AF = AW[i] * np.exp(1j*self.wavenumber*(R_center-R))

            theta_deg = np.round(np.rad2deg(theta),decimals=1)
            thetaf = np.floor(theta_deg).astype(np.int64)
            thetac = np.ceil(theta_deg).astype(np.int64)
            alphat = theta_deg - thetaf

            phi_deg = np.round(np.rad2deg(phi),decimals=1)
            phif = np.floor(phi_deg).astype(np.int64)
            phic = np.ceil(phi_deg).astype(np.int64)
            alphap = phi_deg - phif

            EFtheta = self.element_factor_theta[thetaf,phif] * (1-alphat) * (1-alphap)+ self.element_factor_theta[thetac,phif] * alphat * (1-alphap) \
                    + self.element_factor_theta[thetaf,phic] * (1-alphat) * alphap + self.element_factor_theta[thetac,phic] * alphat * alphap
            EFphi = self.element_factor_phi[thetaf,phif] * (1-alphat) * (1-alphap)+ self.element_factor_phi[thetac,phif] * alphat * (1-alphap) \
                    + self.element_factor_phi[thetaf,phic] * (1-alphat) * alphap + self.element_factor_phi[thetac,phic] * alphat * alphap

            E_theta_sum[i,:,:] = EFtheta*AF # [V/m]
            E_phi_sum[i,:,:] = EFphi*AF # [V/m]

            Ex[i,:,:] = E_theta_sum[i,:,:]*np.cos(theta)*np.cos(phi) - E_phi_sum[i,:,:]*np.sin(phi)
            Ey[i,:,:] = E_theta_sum[i,:,:]*np.cos(theta)*np.sin(phi) + E_phi_sum[i,:,:]*np.cos(phi)
            Ez[i,:,:] = -E_theta_sum[i,:,:]*np.sin(phi)

        Ex_sum = np.sum(Ex,axis=0)
        Ey_sum = np.sum(Ey,axis=0)
        Ez_sum = np.sum(Ez,axis=0)
        self.power_density = self.calibration_factor**2 * (abs(Ex_sum)**2+abs(Ey_sum)**2) * self.input_power / (4*np.pi*(R_center**2)) * 0.1 # mW/cm2
        self.efield_x_sum = Ex_sum
        self.efield_y_sum = Ey_sum
        self.efield_z_sum = Ez_sum

        return self

    def pattern_xzcontour(self, xlim, zlim, y, divnum=1000):
        xrange = np.linspace(xlim[0],xlim[1],num=divnum+1,endpoint=True)
        zrange = np.linspace(zlim[0],zlim[1],num=divnum+1,endpoint=True)
        xrange_mesh, zrange_mesh = np.meshgrid(np.round(xrange,decimals=3), np.round(zrange,decimals=3))
        self.nearfield_x_mesh = xrange_mesh
        self.nearfield_z_mesh = zrange_mesh

        theta_center = np.arctan(np.sqrt((xrange_mesh**2+y**2)/zrange_mesh))
        AW = self.AW.flatten()
        ant_xpos = self.antenna_pos_x_mesh.flatten()
        ant_ypos = self.antenna_pos_y_mesh.flatten()
        R_center = np.sqrt(xrange_mesh**2 + y**2 + zrange_mesh**2)
        element_num = AW.size
        # E_theta_sum = np.zeros((element_num,xrange_mesh.shape[0],xrange_mesh.shape[1]),dtype='complex128')
        # E_phi_sum = np.zeros((element_num,xrange_mesh.shape[0],xrange_mesh.shape[1]),dtype='complex128')
        E_sum = np.zeros((element_num,xrange_mesh.shape[0],xrange_mesh.shape[1]),dtype='complex128')
        for i in range(element_num):
            X = xrange_mesh-ant_xpos[i]
            Y = y-ant_ypos[i]
            R = np.sqrt(X**2 + Y**2 + zrange_mesh**2)
            theta = np.arctan(np.sqrt(X**2+Y**2)/zrange_mesh)
            phi_temp = np.arctan2(Y, X)
            phi = np.where(phi_temp>=0,phi_temp,2*np.pi+phi_temp)
            AF = AW[i] * np.exp(1j*self.wavenumber*(R_center-R))

            theta_deg = np.round(np.rad2deg(theta),decimals=1)
            thetaf = np.floor(theta_deg).astype(np.int64)
            thetac = np.ceil(theta_deg).astype(np.int64)
            alphat = theta_deg - thetaf

            phi_deg = np.round(np.rad2deg(phi),decimals=1)
            phif = np.floor(phi_deg).astype(np.int64)
            phic = np.ceil(phi_deg).astype(np.int64)
            alphap = phi_deg - phif

            # EFtheta = self.element_factor_theta[thetaf,phif] * (1-alphat) * (1-alphap)+ self.element_factor_theta[thetac,phif] * alphat * (1-alphap) \
            #         + self.element_factor_theta[thetaf,phic] * (1-alphat) * alphap + self.element_factor_theta[thetac,phic] * alphat * alphap
            # EFphi = self.element_factor_phi[thetaf,phif] * (1-alphat) * (1-alphap)+ self.element_factor_phi[thetac,phif] * alphat * (1-alphap) \
            #         + self.element_factor_phi[thetaf,phic] * (1-alphat) * alphap + self.element_factor_phi[thetac,phic] * alphat * alphap

            EFabs = self.element_factor_abs[thetaf,phif] * (1-alphat) * (1-alphap)+ self.element_factor_abs[thetac,phif] * alphat * (1-alphap) \
                    + self.element_factor_abs[thetaf,phic] * (1-alphat) * alphap + self.element_factor_abs[thetac,phic] * alphat * alphap


            # E_theta_sum[i,:,:] = EFtheta*AF # [V/m]
            # E_phi_sum[i,:,:] = EFphi*AF # [V/m]
            E_sum[i,:,:] = AF * EFabs # [V/m]


        self.power_density_xz = abs(self.calibration_factor * np.sum(E_sum,axis=0))**2 * self.input_power / (4*np.pi*(R_center**2)) * np.cos(theta_center) * 0.1 # mW/cm2
        Pxz = self.power_density_xz * (14.6*14.6)
        self.power_density_xz_log = 10 * np.log10(Pxz) #dBm

        return self


    def plot_xzcontour(self, xlim, zlim, xticks, zticks, clevel, cticks, xrmin, xrmax, zrmin, zrmax, rline=False):
        plt.rcParams["font.size"] = 30
        fig = plt.figure(figsize=(12,20))
        ax = fig.add_subplot(111)
        levels_ = np.linspace(start=20, stop=45, num=101, endpoint=True)
        ax.set_aspect('equal')
        xmin = xlim[0]
        xmax = xlim[1]
        ax.set_xlim([xmin,xmax])
        zmin = zlim[0]
        zmax = zlim[1]
        ax.set_ylim([zmin,zmax])
        ax.set_xlabel(r'$x\ {\rm [m]}$')
        ax.set_xticks(xticks)
        ax.set_ylabel(r'$z\ {\rm [m]}$')
        ax.set_yticks(zticks)
        if rline:
            ax.hlines(y=[zrmin/1000, zrmax/1000], xmin=xrmin/1000, xmax=xrmax/1000, color='white',linewidth=5)
            ax.vlines(x=[xrmin/1000, xrmax/1000], ymin=zrmin/1000, ymax=zrmax/1000, color='white',linewidth=5)
        im = ax.contourf(self.nearfield_x_mesh,self.nearfield_z_mesh,self.power_density_xz_log,levels=levels_, cmap='jet',extend='both')
        cbar = fig.colorbar(im,ticks=np.arange(20,46,5))
        cbar.set_label(r'Received power ${\rm [dBm]}$')
        self.xzcontour_fig = fig
        return self


    def hpbw_planar(self):
        x = self.nearfield_x_mesh[self.nearfield_y_mesh==0]*1000
        P1_ = self.power_density[self.nearfield_y_mesh==0]
        P1 = P1_/P1_.max()
        P1_hpbw = x[P1>=0.5]
        y = self.nearfield_y_mesh[self.nearfield_x_mesh==0]*1000
        P2_ = self.power_density[self.nearfield_x_mesh==0]
        P2 = P2_/P2_.max()
        P2_hpbw = y[P2>=0.5]
        self.hpbw_data = pd.DataFrame([[P1_hpbw[0],P1_hpbw[-1]],[P2_hpbw[0],P2_hpbw[-1]]],index=['x_hpbw [mm]','y_hpbw [mm]'])
        print('Planar Half Power Beam Width\nX:'+str(P1_hpbw[-1]-P1_hpbw[0])+' mm\nY:'+str(P2_hpbw[-1]-P2_hpbw[0])+' mm')
        return self

    def plot_1Dplane(self, xlim=False, ylim=False, axis="x", offset=0, plot_type='linear', rline=False, xr=0, yr=0):
        plt.rcParams["font.size"] = 24
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        if axis=='x':
            x = self.nearfield_x_mesh[self.nearfield_y_mesh==offset]*1000
            y = self.power_density[self.nearfield_y_mesh==offset]
        elif axis=='y':
            x = self.nearfield_y_mesh[self.nearfield_x_mesh==offset]*1000
            y = self.power_density[self.nearfield_x_mesh==offset]
        if plot_type=='linear':
            ax.plot(x,y)
            ax.set_ylabel(r'$p_{\rm d}\ {\rm [mW/cm^2]}$')
        elif plot_type=='log':
            y_db = 10*np.log10(y/y.max())
            ax.plot(x,y_db)
            ax.set_ylabel(r'$p_{\rm d}\ {\rm [dB]}$')
        if xlim:
            xmin = xlim[0]
            xmax = xlim[1]
            ax.set_xlim([xmin,xmax])
        if ylim:
            ymin = ylim[0]
            ymax = ylim[1]
            ax.set_ylim([ymin,ymax])
        ax.set_xlabel(axis+' [mm]')
        if rline:
            if axis == 'x':
                ax.vlines([-xr/2,xr/2],ymin,ymax,colors='red')
            if axis == 'y':
                ax.vlines([-yr/2,yr/2],ymin,ymax,colors='red')
        self.nf_1dplot_fig = fig

        return self

    def divisional_reception(self, xr, yr, xdiv, ydiv):
        index_x = (self.nearfield_x_mesh[0,:]>=-xr/2000)&(self.nearfield_x_mesh[0,:]<=xr/2000)
        index_y = (self.nearfield_y_mesh[:,0]>=-yr/2000)&(self.nearfield_y_mesh[:,0]<=yr/2000)
        total_power = self.power_density[np.ix_(index_y,index_x)]
        recept_power = np.zeros((ydiv,xdiv))
        unit_area = (self.nearfield_x_mesh[0,1] - self.nearfield_x_mesh[0,0]) * (self.nearfield_y_mesh[1,0] - self.nearfield_y_mesh[0,0])
        for i in range(ydiv):
            for j in range(xdiv):
                x_sec_size = total_power.shape[1]//xdiv
                y_sec_size = total_power.shape[0]//ydiv
                recept_power[i,j] = np.sum(total_power[y_sec_size*i:y_sec_size*(i+1),x_sec_size*j:x_sec_size*(j+1)]*unit_area*10)
        return recept_power

    def plot_1Dplane_angle(self, xlim=False, ylim=False, axis='y', offset=0, rline=False):
        plt.rcParams["font.size"] = 24
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        if axis=='x':
            x = self.nearfield_x_mesh[self.nearfield_y_mesh==offset]*1000
            y = self.efield_phase[self.nearfield_y_mesh==offset]
        elif axis=='y':
            x = self.nearfield_y_mesh[self.nearfield_x_mesh==offset]*1000
            y = self.efield_phase[self.nearfield_x_mesh==offset]

        ax.plot(x,y)
        ax.set_ylabel('Phase [deg.]')

        if xlim:
            xmin = xlim[0]*1000
            xmax = xlim[1]*1000
            ax.set_xlim([xmin,xmax])
        if ylim:
            ymin = ylim[0]*1000
            ymax = ylim[1]*1000
            ax.set_ylim([ymin,ymax])
        ax.set_xlabel(r'$x\ {\rm [mm]}$')
        if rline:
            ax.vlines([-0.1462/2,0.1462/2],ymin,ymax,colors='red')
        self.nf_1dplot_phase_fig = fig

        return self

    def include_rect_efficiency(self,rect_file,recept_power):
        today_ = datetime.datetime.today()
        today = today_.strftime('%y%m%d')
        rect_data = pd.read_csv('data/rectifier/'+rect_file+'.csv')
        pin_ref = rect_data['Pin [W]'].values
        eff_ref = rect_data['Efficiency [%]'].values
        rect_eff = np.zeros(recept_power.shape)
        for i in range(recept_power.shape[0]):
            for j in range(recept_power.shape[1]):
                pin_floor = (pin_ref[pin_ref<recept_power[i,j]])[-1]
                pin_ceil = (pin_ref[pin_ref>=recept_power[i,j]])[0]
                eff_floor = (eff_ref[pin_ref<recept_power[i,j]])[-1]
                eff_ceil = (eff_ref[pin_ref>=recept_power[i,j]])[0]
                rect_eff[i,j] = (recept_power[i,j]-pin_floor)/(pin_ceil-pin_floor) * eff_ceil\
                + (pin_ceil-recept_power[i,j])/(pin_ceil-pin_floor) * eff_floor

        rect_out = recept_power * rect_eff/100
        os.makedirs('plot/2D/'+self.excitation_manner+'/'+today,exist_ok='True')
        pd.DataFrame(rect_eff).to_csv('plot/2D/'+self.excitation_manner+'/'+today+'/rectifier_efficiency.csv')
        pd.DataFrame(rect_out).to_csv('plot/2D/'+self.excitation_manner+'/'+today+'/rectifier_output.csv')
        total_out = np.sum(rect_out)
        total_in = np.sum(recept_power)
        total_rect_eff = total_out/total_in*100
        total_wpt_eff = total_out/self.input_power*100
        total_profile = pd.DataFrame(np.vstack([self.input_power,total_in,total_out,total_rect_eff,total_wpt_eff]),\
        index=['Total Input [W]','Rectenna Input [W]','Rectenna Output [W]','Total Rectification Efficiency [%]','Total WPT Efficiency [%]'])
        total_profile.to_csv('plot/2D/'+self.excitation_manner+'/'+today+'/rectenna_profile.csv',header=None)
        return rect_eff



    def save_results(self,save_ff,save_nf,save_aw,save_awp,save_ef):
        today_ = datetime.datetime.today()
        today = today_.strftime('%y%m%d')
        if save_ff:
            os.makedirs('plot/farfield/'+self.excitation_manner+'/'+today,exist_ok=True)
            self.farfield_fig.savefig('plot/farfield/'+self.excitation_manner+'/'+today+'/'+self.fn3+'.png')
            self.farfield_dataframe.to_excel('plot/farfield/'+self.excitation_manner+'/'+today+'/'+self.fn3+'.xlsx',header=None)
        if save_nf:
            os.makedirs('plot/2D/'+self.excitation_manner+'/'+today,exist_ok=True)
            os.makedirs('plot/1D/'+self.excitation_manner+'/'+today,exist_ok=True)
            self.nf_2dcontour_fig.savefig('plot/2D/'+self.excitation_manner+'/'+today+'/'+self.fn1+'.png')
            self.nf_1dplot_fig.savefig('plot/1D/'+self.excitation_manner+'/'+today+'/'+self.fn2+'.png')
            self.hpbw_data.to_excel('plot/2D/'+self.excitation_manner+'/'+today+'/'+self.fn7+'.xlsx',header=None)
        if save_aw:
            os.makedirs('weight/'+self.excitation_manner+'/'+today,exist_ok=True)
            self.aw_profile.to_excel('weight/'+self.excitation_manner+'/'+today+'/'+self.fn4+'.xlsx',index=None)
        if save_awp:
            os.makedirs('weight/'+self.excitation_manner+'/'+today,exist_ok=True)
            self.aw_amp_figure.savefig('weight/'+self.excitation_manner+'/'+today+'/'+self.fn5+'.png')
            self.aw_phase_figure.savefig('weight/'+self.excitation_manner+'/'+today+'/'+self.fn6+'.png')
        if save_ef:
            os.makedirs('plot/2D/'+self.excitation_manner+'/'+today,exist_ok=True)
            self.efficiency_profile.to_excel('plot/2D/'+self.excitation_manner+'/'+today+'/'+self.fn8+'.xlsx',header=None)


    def comb_alignment(self, ar1, ar2):
        Xpos1 = ar1.antenna_pos_x_mesh.flatten()
        Ypos1 = ar1.antenna_pos_y_mesh.flatten()
        Xpos2 = ar2.antenna_pos_x_mesh.flatten()
        Ypos2 = ar2.antenna_pos_y_mesh.flatten()
        self.antenna_pos_x_mesh = np.block([Xpos1,Xpos2])
        self.antenna_pos_y_mesh = np.block([Ypos1,Ypos2])

        AW1 = ar1.array_weight.flatten()
        AW2 = ar2.array_weight.flatten()
        self.array_weight = np.block([AW1,AW2])

        return self
