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
        self.waypoints = None
        self.fig, self.axs = plt.subplots(1,4,figsize=(10,4))
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

    def onclick(self,event):
        if event.key == 'control':
            x = float(event.xdata)
            y = float(event.ydata)
            p = np.array([x,y]).reshape(1,-1)
            self.waypoints = p if self.waypoints is None else np.vstack((self.waypoints,p))
            dt = 0.01
            t = dt*self.find_closest_index(self.mu,p)
            self.__logger.info('Inserting new point x=%.2f, y=%.2f, t=%.2f' % (event.xdata, event.ydata, t))
            time = dt*np.arange(1,201).reshape(1,-1)
            var = np.eye(2)*1e-6
            self.kmp.set_waypoint(t,p,var)
            mu_kmp, sigma_kmp = self.kmp.predict(time)
            self.axs[3].cla()
            self.axs[3].plot(self.mu[0],self.mu[1],color='gray')
            self.plot_gmm(self.axs[3], mu_kmp, sigma_kmp)
            self.axs[3].scatter(self.waypoints[:,0],self.waypoints[:,1])
            self.axs[3].set_axisbelow(True)
            self.axs[3].grid(visible=True)
            
    def plot_demos(self, ax):
        for h in range(self.gmm.n_demos_):
            coords = self.demos[h]['pos']
            ax.plot(coords[0,:],coords[1,:],color='green',zorder=0)
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_axisbelow(True)
        ax.grid(visible=True)

    def plot_gmm(self, ax, mu, sigma):
        for c in range(sigma.shape[2]):
            cov_pos = sigma[:,:,c]
            e_vals, e_vecs = np.linalg.eig(cov_pos)
            major_axis = 2 * np.sqrt(e_vals[0]) * e_vecs[:, 0]
            minor_axis = 2 * np.sqrt(e_vals[1]) * e_vecs[:, 1]
            angle = np.degrees(np.arctan2(major_axis[1], major_axis[0]))
            pos = [mu[0,c],mu[1,c]] 
            width = np.linalg.norm(major_axis)
            height = np.linalg.norm(minor_axis)
            ellipse = Ellipse(xy=pos, width=width, height=height, angle=angle, facecolor='orange', alpha=0.5,zorder=1)
            ax.add_artist(ellipse)
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        if mu.shape[1] == self.gmm.n_components_:
            ax.scatter(mu[0],mu[1],zorder=3)
        else:
            ax.plot(mu[0],mu[1],color='green')
        ax.set_axisbelow(True)
        ax.grid(visible=True)

    def update(self, event):
        # Clear previous plots
        self.axs[0].cla()
        self.axs[1].cla()
        self.axs[2].cla()
        self.axs[3].cla()
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
                vel = np.concatenate([d['vel'] for d in self.demos], axis=1)
                N = pos.shape[1] # Length of each demonstration
                self.gmm = GaussianMixtureModel(n_components=C,n_demos=H)
                # Input: time, output: position
                dt = 0.01
                X = dt*np.tile(np.linspace(1,int(N/H),int(N/H)),H).reshape(1,-1)
                Y = pos
                self.gmm.fit(X,Y)
                # Plot the GMM
                self.plot_gmm(self.axs[1],self.gmm.means_[1:,:],self.gmm.covariances_[1:,1:,:])
                # Compute the reference trajectory with GMR
                self.mu, self.sigma = self.gmm.predict(dt*np.arange(int(N/H)).reshape(1,-1))
                # Plot the GMR
                self.plot_gmm(self.axs[2], self.mu, self.sigma)
                # Plot the demonstration database
                self.plot_demos(self.axs[0])
                self.plot_demos(self.axs[1])
                # Do KMP
                self.axs[3].plot(self.mu[0],self.mu[1],color='gray')
                time = dt*np.arange(1,int(N/H)+1).reshape(1,-1)
                self.kmp = KMP()
                self.kmp.fit(time, self.mu, self.sigma)
                mu_kmp, sigma_kmp = self.kmp.predict(time)
                self.plot_gmm(self.axs[3], mu_kmp, sigma_kmp)
                self.axs[0].set_title('GMM - initial')
                self.axs[1].set_title('GMM - EM')
                self.axs[2].set_title('GMR')
                self.axs[3].set_title('KMP')