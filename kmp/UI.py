#!/usr/bin/env python3

import logging
import matplotlib.pyplot as plt
import numpy as np
import os

from kmp import utils
from kmp.mixture import GaussianMixtureModel
from kmp.model import KMP
from matplotlib.patches import Ellipse
from matplotlib.widgets import Button, TextBox
    
class UI:
    def __init__(self) -> None:
        self.__logger = logging.getLogger(__name__)
        plt.ion()
        self.waypoints1 = None
        self.waypoints2 = None
        self.fig, self.axs = plt.subplots(2,3,figsize=(10,8))
        self.fig.subplots_adjust(bottom=0.2)
        axbtn = self.fig.add_axes([0.81, 0.05, 0.1, 0.075])
        axC = self.fig.add_axes([0.7, 0.05, 0.1, 0.075])
        axletter = self.fig.add_axes([0.58, 0.05, 0.1, 0.075])
        self.btn = Button(axbtn, 'Update')
        self.btn.on_clicked(self.update)
        self.Ctxt = TextBox(axC,'C','8')
        self.lettertxt = TextBox(axletter,'letter','G')
        self.update(0)
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.show() 
        
    def find_closest_index(self, list, val):
        smallest_dist = np.inf
        for i,l in enumerate(list.T):
            dist = np.linalg.norm(val-l)
            if dist < smallest_dist:
                closest_index = i
                smallest_dist = dist
        return closest_index

    # Handles the addition of waypoints and the update of the corresponding KMP and of the superposition
    def onclick(self,event):
        row, col = np.unravel_index(list(self.axs.flat).index(event.inaxes),(2,3))
        if event.key == 'control' and ([row,col]==[1,0] or [row,col]==[1,1]):
            x = float(event.xdata)
            y = float(event.ydata)
            p = np.array([x,y]).reshape(1,-1)
            dt = self.kmp_dt
            id = self.find_closest_index(self.mu[:2,:],p)
            t = dt*(id+1)
            self.__logger.info('Inserting new point x=%.2f, y=%.2f, t=%.2f' % (event.xdata, event.ydata, t))
            time = dt*np.arange(1,201).reshape(1,-1)
            var = np.eye(2)*1e-6
            p = np.array([x,y]).reshape(1,-1)
            self.axs[row, col].cla()
            self.axs[row, col].plot(self.mu[0],self.mu[1],color='gray')
            if col == 0:
                self.kmp1.set_waypoint([t],[p],[var])
                self.waypoints1 = p[:2] if self.waypoints1 is None else np.vstack((self.waypoints1,p[:2]))
                self.mu_kmp1, self.sigma_kmp1 = self.kmp1.predict(time)
                self.plot_gmm(self.axs[row,col], self.mu_kmp1, self.sigma_kmp1)
                self.axs[row, col].scatter(self.waypoints1[:,0],self.waypoints1[:,1])
            else:
                self.kmp2.set_waypoint([t],[p],[var])
                self.waypoints2 = p[:2] if self.waypoints2 is None else np.vstack((self.waypoints2,p[:2]))
                self.mu_kmp2, self.sigma_kmp2 = self.kmp2.predict(time)
                self.plot_gmm(self.axs[row,col], self.mu_kmp2, self.sigma_kmp2)
                self.axs[row, col].scatter(self.waypoints2[:,0],self.waypoints2[:,1])
            self.axs[row, col].set_axisbelow(True)
            self.axs[row, col].grid(visible=True)
            self.axs[row, col].set_title('KMP')
            self.kmp_sup.fit(time,[self.mu_kmp1, self.mu_kmp2], [self.sigma_kmp1, self.sigma_kmp2])
            mu_kmp_sup, sigma_kmp_sup = self.kmp_sup.predict(time)
            self.axs[1,2].cla()
            self.axs[1,2].plot(self.mu[0],self.mu[1],color='gray')
            self.plot_gmm(self.axs[1,2], mu_kmp_sup, sigma_kmp_sup)
            self.axs[1,2].set_axisbelow(True)
            self.axs[1,2].grid(visible=True)
            self.axs[1,2].set_title('KMP - superposition')
            
    # Plots the positions in the demonstration database
    def plot_demos(self, ax):
        for h in range(self.gmm.n_demos_):
            coords = self.demos[h]['pos']
            ax.plot(coords[0,:],coords[1,:],color='green',zorder=0)
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_axisbelow(True)
        ax.grid(visible=True)

    # Plots a trajectory defined by a mean vector array and a covariance matrix array
    def plot_gmm(self, ax, mu, sigma, color='green'):
        if sigma is not None:
            for c in range(sigma.shape[2]):
                cov_pos = sigma[:,:,c]
                e_vals, e_vecs = np.linalg.eig(cov_pos)
                major_axis = 2 * np.sqrt(e_vals[0]) * e_vecs[:, 0]
                minor_axis = 2 * np.sqrt(e_vals[1]) * e_vecs[:, 1]
                angle = np.degrees(np.arctan2(major_axis[1], major_axis[0]))
                pos = [mu[0,c],mu[1,c]] 
                width = np.linalg.norm(major_axis)
                height = np.linalg.norm(minor_axis)
                ellipse = Ellipse(xy=pos, width=width, height=height, angle=angle, facecolor='orange', alpha=0.6,zorder=1)
                ax.add_artist(ellipse)
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        if mu.shape[1] == self.gmm.n_components_:
            ax.scatter(mu[0],mu[1],zorder=3)
        else:
            ax.plot(mu[0],mu[1],color=color)
        ax.set_axisbelow(True)
        ax.grid(visible=True)

    def p1(self, s):
        # Defines the priority for the first trajectory in the superposition
        # NB: p1 + p2 = 1
        return np.exp(-s)
    
    def p2(self, s):
        # Defines the priority for the second trajectory in the superposition
        # NB: p1 + p2 = 1
        return 1-np.exp(-s)

    # Handles the generation of the various plots
    def update(self, event):
        # Clear previous plots
        self.axs[0,0].cla()
        self.axs[0,1].cla()
        self.axs[0,2].cla()
        self.axs[1,0].cla()
        self.axs[1,1].cla()
        self.axs[1,2].cla()
        self.waypoints1 = None
        self.waypoints2 = None
        # Extract the demonstrations for the given letter
        if str.isnumeric(self.Ctxt.text):
            C = int(self.Ctxt.text)
            letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
                    'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
            letter = str.upper(self.lettertxt.text)
            if letter in letters:
                # Reinit the model
                path = os.getcwd() + '/2Dletters/' + letter + '.mat'
                H = 5 # Number of demonstrations
                self.demos = utils.read_struct(path,max_cell=H)
                pos = np.concatenate([d['pos'] for d in self.demos], axis=1)
                N = pos.shape[1] # Length of each demonstration
                self.gmm = GaussianMixtureModel(n_components=C,n_demos=H)
                # Input: time, output: position
                dt = 0.01
                X = dt*np.tile(np.linspace(1,int(N/H),int(N/H)),H).reshape(1,-1)
                Y = pos
                self.gmm.fit(X,Y)
                # Plot the GMM
                self.plot_gmm(self.axs[0,1],self.gmm.means_[1:3,:],self.gmm.covariances_[1:3,1:3,:])
                # Compute the reference trajectory with GMR
                self.mu, self.sigma = self.gmm.predict(dt*np.arange(int(N/H)).reshape(1,-1))
                # Plot the GMR
                self.plot_gmm(self.axs[0,2], self.mu[:2,:], self.sigma[:2,:2,:])
                # Plot the demonstration database
                self.plot_demos(self.axs[0,0])
                self.plot_demos(self.axs[0,1])
                # Input vector for KMP
                self.kmp_dt = 0.01
                time = self.kmp_dt*np.arange(1,int(N/H)+1).reshape(1,-1)
                # Set up the first KMP
                self.kmp1 = KMP()
                self.kmp1.fit(time, self.mu, self.sigma)
                self.mu_kmp1, self.sigma_kmp1 = self.kmp1.predict(time)
                self.axs[1,0].plot(self.mu[0],self.mu[1],color='gray')
                self.plot_gmm(self.axs[1,0], self.mu_kmp1[:2,:], self.sigma_kmp1[:2,:2,:])
                # Set up the second KMP
                self.kmp2 = KMP()
                self.kmp2.fit(time,self.mu,self.sigma)
                self.mu_kmp2, self.sigma_kmp2 = self.kmp2.predict(0.01*np.arange(1,int(N/H)+1).reshape(1,-1))
                self.axs[1,1].plot(self.mu[0],self.mu[1],color='gray')
                self.plot_gmm(self.axs[1,1], self.mu_kmp2[:2,:], self.sigma_kmp2[:2,:2,:])
                # Setup the superposition KMP
                self.kmp_sup = KMP(priorities=[self.p1,self.p2])
                self.kmp_sup.fit(time,[self.mu_kmp1, self.mu_kmp2], [self.sigma_kmp1, self.sigma_kmp2])
                mu_kmp_sup, sigma_kmp_sup = self.kmp_sup.predict(time)
                self.axs[1,2].plot(self.mu[0],self.mu[1],color='gray')
                self.plot_gmm(self.axs[1,2], mu_kmp_sup[:2,:], sigma_kmp_sup[:2,:2,:])
                self.axs[0,0].set_title('Demonstration database')
                self.axs[0,1].set_title('GMM')
                self.axs[0,2].set_title('GMR')
                self.axs[1,0].set_title('KMP')
                self.axs[1,1].set_title('KMP')
                self.axs[1,2].set_title('KMP - superposition')
