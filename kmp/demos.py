#!/usr/bin/env python3

import logging
import matplotlib.pyplot as plt
import numpy as np
import os

from kmp import utils
from kmp.mixture import GaussianMixtureModel
from kmp.model import KMP
from kmp.types.quaternion import quaternion
from matplotlib.patches import Ellipse
from matplotlib.widgets import Button, TextBox
    

class demo1:
    """Kernelized Movement Primitives: learning and adaptation. Only position is considered, to 
    allow for dynamic addition of points to the trajectory.
    """

    def __init__(self) -> None:
        self.__logger = logging.getLogger(__name__)
        # Turn on interactive figures
        plt.ion()
        # Set up the plots and buttons
        self.fig, self.axs = plt.subplots(1,4,figsize=(10,8))
        self.fig.subplots_adjust(bottom=0.2)
        axbtn = self.fig.add_axes([0.81, 0.05, 0.1, 0.075])
        axC = self.fig.add_axes([0.7, 0.05, 0.1, 0.075])
        axletter = self.fig.add_axes([0.58, 0.05, 0.1, 0.075])
        self.btn = Button(axbtn, 'Update')
        self.btn.on_clicked(self.update)
        self.Ctxt = TextBox(axC,'C','8')
        self.lettertxt = TextBox(axletter,'letter','G')
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        # Keep track of the added waypoints for KMP
        self.waypoints = None
        # Perform the first update and show the results
        self.update(0)
        self.fig.show() 

    def plot_demos(self, ax, subsample) -> None:
        """Plots the demonstration database on the given axis.
        """
        for h in range(self.gmm.n_demos):
            coords = self.demos[h]['pos']
            ax.plot(coords[0,::subsample],coords[1,::subsample],color='green',zorder=0)
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_axisbelow(True)
        ax.grid(visible=True)

    def plot_gmm(self, ax, mu, sigma, color='green') -> None:
        """Plots a trajectory defined point by point by a mean and a covariance matrix on the given 
        axis.
        """
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
        if mu.shape[1] == self.gmm.n_components:
            ax.scatter(mu[0],mu[1],zorder=3)
        else:
            ax.plot(mu[0,:],mu[1,:],color=color)
        ax.set_axisbelow(True)
        ax.grid(visible=True)

    def onclick(self, event):
        """Handle clicks on KMP's axis to carry out adaptation
        """
        axes_title = event.inaxes.axes.title._text
        if event.key == 'control' and axes_title == "KMP":
            x = float(event.xdata)
            y = float(event.ydata)
            p = np.array([x,y]).reshape(1,-1)
            dt = self.kmp_dt
            id = utils.find_closest_index(self.mu[:2,:],p)
            t = dt*(id+1)
            self.__logger.info('Inserting new point x=%.2f, y=%.2f, t=%.2f' % (event.xdata, event.ydata, t))
            time = self.kmp_dt*np.arange(1,int(self.N/self.H)+1).reshape(1,-1)
            var = np.eye(2)*1e-4
            p = np.array([x,y]).reshape(1,-1)
            # Clear the plot, plot again GMR's reference
            self.axs[3].cla()
            self.axs[3].plot(self.mu[0],self.mu[1],color='gray')
            # Update the prediction with KMP
            self.kmp.set_waypoint([t],[p],[var])
            self.waypoints = p[:2] if self.waypoints is None else np.vstack((self.waypoints,p[:2]))
            self.mu_kmp, self.sigma_kmp = self.kmp.predict(time)
            # Plot the result, including the location of the waypoint
            self.plot_gmm(self.axs[3], self.mu_kmp, self.sigma_kmp)
            self.axs[3].scatter(self.waypoints[:,0],self.waypoints[:,1])
            # Make the plot nicer looking
            self.axs[3].set_axisbelow(True)
            self.axs[3].grid(visible=True)
            self.axs[3].set_title('KMP')

    def update(self, event):
        # Clear previous plots and the waypoint arrays
        self.axs[0].cla()
        self.axs[1].cla()
        self.axs[2].cla()
        self.axs[3].cla()
        self.waypoints = None
        # Extract the demonstrations for the given letter
        if str.isnumeric(self.Ctxt.text): 
            C = int(self.Ctxt.text) # Number of Gaussian components
            letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
                    'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
            letter = str.upper(self.lettertxt.text)
            if letter in letters:

                # Load the letter data from the MATLAB struct
                path = os.getcwd() + '/2Dletters/' + letter + '.mat'
                self.demos = utils.read_struct(path,max_cell=5)
                # Extract the position data
                pos = np.concatenate([d['pos'] for d in self.demos], axis=1)
                # Reducing sample size affects sligthly accuracy, but greatly improves performance
                subsample = 4 # Keep 25% of samples
                pos = pos[:, ::subsample]
                self.N = pos.shape[1] # Length of each demonstration
                self.H = 5 # Number of demonstrations

                # Prepare the data for the GMM and GMR (input: time, output: position)
                dt = 0.01
                time = dt*np.tile(np.linspace(1,int(self.N/self.H),int(self.N/self.H)),self.H).reshape(1,-1)
                X = np.vstack((time, pos)).T # Transpose to have shape (n_features, n_samples)
                x_gmr = dt*np.arange(int(self.N/self.H)).reshape(1,-1) # Input for GMR

                # Train the GMM and make a prediction using GMR
                self.gmm = GaussianMixtureModel(n_components=C, diag_reg_factor=1e-6)
                self.gmm.fit(X)
                self.mu, self.sigma = self.gmm.predict(x_gmr)

                # Plot the results
                self.plot_gmm(self.axs[1], self.gmm.means[1:3,:], self.gmm.covariances[1:3,1:3,:])
                self.plot_gmm(self.axs[2], self.mu[:2,:], self.sigma[:2,:2,:])
                # Plot the demonstration database
                self.plot_demos(self.axs[0], subsample)
                self.plot_demos(self.axs[1], subsample)

                # Prepare the data for KMP
                self.kmp_dt = 0.01
                time = self.kmp_dt*np.arange(1,int(self.N/self.H)+1).reshape(1,-1)

                # Set up KMP
                self.kmp = KMP(l=0.5, alpha=40, sigma_f=200, time_driven_kernel=False)
                self.kmp.fit(time, self.mu, self.sigma)
                self.mu_kmp, self.sigma_kmp = self.kmp.predict(time)
                
                # Plot KMP's results
                self.axs[3].plot(self.mu[0],self.mu[1],color='gray')
                self.plot_gmm(self.axs[3], self.mu_kmp[:2,:], self.sigma_kmp[:2,:2,:])
                
                self.axs[0].set_title('Demonstration database')
                self.axs[1].set_title('GMM')
                self.axs[2].set_title('GMR')
                self.axs[3].set_title('KMP')

class demo2:
    """Generalized orientation learning with KMP."""

    
    def __init__(self) -> None:
        self.__logger = logging.getLogger(__name__)

        # Load the pose data or generate it if it wasn't before
        dataset_path = os.path.join(os.getcwd() + '/quaternion_trajectories/pose_data.npy')
        if not os.path.isfile(dataset_path):
            path = os.path.join(os.getcwd() + '/quaternion_trajectories/')
            self.demos = utils.create_dataset(path,subsample=50)
        else:
            self.demos = np.load(dataset_path, allow_pickle=True)

        # Plot the demonstrations and set up the plots for the rest
        self.fig, self.axs = plt.subplots(3,3,figsize=(10,8))
        self.axs[0,0].set_title('Real data')
        self.axs[0,1].set_title('Learning')
        self.axs[0,2].set_title('Adaptation')
        demo_num = 6
        demo_len = int(len(self.demos)/demo_num)
        for j in range(demo_num):
            time = [s.time for i,s in enumerate(self.demos) if i >= j*demo_len and i < (j+1)*demo_len]
            x = [s.pose[0] for i,s in enumerate(self.demos) if i >= j*demo_len and i < (j+1)*demo_len]
            y = [s.pose[1] for i,s in enumerate(self.demos) if i >= j*demo_len and i < (j+1)*demo_len]
            z = [s.pose[2] for i,s in enumerate(self.demos) if i >= j*demo_len and i < (j+1)*demo_len]
            q1 = [s.quat_eucl[0] for i,s in enumerate(self.demos) if i >= j*demo_len and i < (j+1)*demo_len]
            q2 = [s.quat_eucl[1] for i,s in enumerate(self.demos) if i >= j*demo_len and i < (j+1)*demo_len]
            q3 = [s.quat_eucl[2] for i,s in enumerate(self.demos) if i >= j*demo_len and i < (j+1)*demo_len]
            qx = [s.quat[1] for i,s in enumerate(self.demos) if i >= j*demo_len and i < (j+1)*demo_len]
            qy = [s.quat[2] for i,s in enumerate(self.demos) if i >= j*demo_len and i < (j+1)*demo_len]
            qz = [s.quat[3] for i,s in enumerate(self.demos) if i >= j*demo_len and i < (j+1)*demo_len]
            qs = [s.quat[0] for i,s in enumerate(self.demos) if i >= j*demo_len and i < (j+1)*demo_len]
            if j == 0:
                self.axs[0,0].plot(time,x,color='red',label='x')
                self.axs[0,0].plot(time,y,color='green',label='y')
                self.axs[0,0].plot(time,z,color='blue',label='z')
                self.axs[1,0].plot(time,qs,color='red',label='qs')
                self.axs[1,0].plot(time,qx,color='green',label='qx')
                self.axs[1,0].plot(time,qy,color='blue',label='qy')
                self.axs[1,0].plot(time,qz,color='purple',label='qz')
                self.axs[2,0].plot(time,q1,color='red',label='q1')
                self.axs[2,0].plot(time,q2,color='green',label='q2')
                self.axs[2,0].plot(time,q3,color='blue',label='q3')
            else:
                self.axs[0,0].plot(time,x,color='red')
                self.axs[0,0].plot(time,y,color='green')
                self.axs[0,0].plot(time,z,color='blue')
                self.axs[1,0].plot(time,qs,color='red')
                self.axs[1,0].plot(time,qx,color='green')
                self.axs[1,0].plot(time,qy,color='blue')
                self.axs[1,0].plot(time,qz,color='purple')
                self.axs[2,0].plot(time,q1,color='red')
                self.axs[2,0].plot(time,q2,color='green')
                self.axs[2,0].plot(time,q3,color='blue')
        self.axs[0,0].legend()
        self.axs[1,0].legend()
        self.axs[2,0].legend()

        # GMM and KMP on the position
        gmm = GaussianMixtureModel(n_demos=demo_num)
        time = np.array([s.time for s in self.demos]).reshape(1,-1)
        pos = np.vstack([s.pose[:3] for s in self.demos]).T
        X = np.vstack((time, pos)).T # Transpose to have shape (n_features, n_samples)
        gmm.fit(X)
        time_single = np.array([s.time for i,s in enumerate(self.demos) if i < demo_len]).reshape(1,-1)
        mu_pos, sigma_pos = gmm.predict(time_single)

        # Learning
        kmp_pos = KMP(l=0.5, alpha=60, sigma_f=4, time_driven_kernel=False)
        kmp_pos.fit(time_single, mu_pos, sigma_pos)
        mu_pos_kmp, sigma_pos_kmp = kmp_pos.predict(time_single)
        self.plot_mean(self.axs[0,1],time_single.T,mu_pos_kmp[:3])

        # Adaptation
        # Note that I use another KMP instance here, but just for graphical purposes. kmp_pos could have been used
        kmp_pos_ad = KMP(l=0.5, alpha=60, sigma_f=4, time_driven_kernel=False)
        kmp_pos_ad.fit(time_single, mu_pos, sigma_pos)
        waypoint = [0.0,0.85,0.3]
        kmp_pos_ad.set_waypoint([1],np.array(waypoint).reshape(1,-1),np.eye(3)*1e-6)
        self.axs[0,2].scatter([1,1,1],waypoint[:3])
        mu_pos_kmp_ad, sigma_pos_kmp_ad = kmp_pos_ad.predict(time_single)
        self.plot_mean(self.axs[0,2],time_single.T,mu_pos_kmp_ad[:3])

        # GMM on the quaternions
        gmm_quat = GaussianMixtureModel(n_demos=demo_num)
        quats_eucl = np.vstack([s.quat_eucl for s in self.demos]).T
        X = np.vstack((time, quats_eucl)).T # Transpose to have shape (n_features, n_samples)
        gmm_quat.fit(X)
        mu_quat, sigma_quat = gmm_quat.predict(time_single)

        # Learning
        kmp_quat = KMP(l=0.5, alpha=60, sigma_f=4, time_driven_kernel=False)
        kmp_quat.fit(time_single, mu_quat, sigma_quat)
        mu_quat_kmp, sigma_quat_kmp = kmp_quat.predict(time_single)
        self.plot_mean(self.axs[2,1],time_single.T,mu_quat_kmp)
        # Recover the auxiliary quaternion
        quats = np.vstack([s.quat for s in self.demos])
        qa = quats[0][0]
        # Project the results back into quaternion space
        kmp_quats = np.vstack((mu_quat_kmp[:3,:],np.zeros_like(mu_quat_kmp[0,:])))
        for i in range(mu_quat_kmp.shape[1]):
            tmp = quaternion.exp(kmp_quats[:3,i])
            kmp_quats[:,i] = (tmp*qa).as_array()
        self.plot_mean(self.axs[1,1],time_single.T,kmp_quats)

        # Adaptation
        kmp_quat_ad = KMP(l=0.5, alpha=60, sigma_f=4, time_driven_kernel=False)
        kmp_quat_ad.fit(time_single,mu_quat,sigma_quat)
        # Project to euclidean space
        des_quat = quaternion(-0.2,np.array([0.8,-0.7,-1.6]))
        self.axs[1,2].scatter([1,1,1,1],des_quat.as_array(),marker='+')
        waypoint = (des_quat*qa).log()
        kmp_quat_ad.set_waypoint([1],waypoint.reshape(1,-1),np.eye(3)*1e-6)
        self.axs[2,2].scatter([1,1,1],waypoint[:3],marker='+')
        mu_quat_kmp_ad, sigma_quat_kmp_ad = kmp_quat_ad.predict(time_single)
        self.plot_mean(self.axs[2,2],time_single.T,mu_quat_kmp_ad)
        # Project the results back into quaternion space
        kmp_quats = np.vstack((mu_quat_kmp_ad[:3,:],np.zeros_like(mu_quat_kmp_ad[0,:])))
        for i in range(kmp_quats.shape[1]):
            tmp = quaternion.exp(kmp_quats[:3,i])
            kmp_quats[:,i] = (tmp*qa).as_array()
        self.plot_mean(self.axs[1,2],time_single.T,kmp_quats)

        # Make the plots look nicer
        for row in self.axs: 
            for ax in row:
                ax.set_xlim(0,time[0,-1])
                ax.set_ylim(-1.5,1.5)
                ax.grid()
                ax.set_axisbelow(True)
        self.fig.show()

    def plot_mean(self, ax, time, mu):
        ax.plot(time, mu[0,:],color='red')
        ax.plot(time, mu[1,:],color='green')
        ax.plot(time, mu[2,:],color='blue')
        if mu.shape[0] == 4:
            ax.plot(time, mu[3,:],color='purple')

class demo3:
    # KMP with linear velocity learning and adaptation.
    def __init__(self) -> None:
        self.__logger = logging.getLogger(__name__)
        plt.ion()
        self.fig, self.axs = plt.subplots(1,5,figsize=(16,4))
        # Position/velocity KMP
        path = os.getcwd() + '/2Dletters/G.mat'
        H = 5 # Number of demonstrations
        self.demos = utils.read_struct(path,max_cell=H)
        pos = np.concatenate([d['pos'] for d in self.demos], axis=1)
        vel = np.concatenate([d['vel'] for d in self.demos], axis=1)
        N = pos.shape[1] # Length of each demonstration
        self.gmm = GaussianMixtureModel(n_demos=H)
        # Input: time, output: position/velocity
        dt = 0.01
        time = dt*np.tile(np.linspace(1,int(N/H),int(N/H)),H).reshape(1,-1)
        X = np.vstack((time,pos,vel)).T
        self.gmm.fit(X)
        # Compute the reference trajectory with GMR
        self.mu, self.sigma = self.gmm.predict(dt*np.arange(int(N/H)).reshape(1,-1))
        # Plot the GMR
        self.axs[0].plot(self.mu[0,:],self.mu[1,:],color="grey")
        # Input vector for KMP
        self.kmp_dt = 0.01
        time = self.kmp_dt*np.arange(1,int(N/H)+1).reshape(1,-1)
        # Set up the first KMP
        self.kmp_pos = KMP(l=1,sigma_f=6)
        t = [0.01,0.25,1.2,2]
        p1 = np.array([8, 10, -50, 0]).reshape(1,-1)
        p2 = np.array([-1, 6, -25, -40]).reshape(1,-1)
        p3 = np.array([8, -4, 30, 10]).reshape(1,-1)
        p4 = np.array([-3, 1, -10, 3]).reshape(1,-1)
        p = [p1,p2,p3,p4]
        var = np.eye(4)*1e-6
        self.kmp_pos.fit(time, self.mu, self.sigma)
        self.kmp_pos.set_waypoint(t, p, [var,var,var,var])
        self.kmp_pos, _ = self.kmp_pos.predict(time)
        self.axs[0].plot(self.kmp_pos[0,:],self.kmp_pos[1,:],color="green")
        time = np.reshape(time, (200))
        self.axs[1].plot(time,self.kmp_pos[0,:],color="green")
        self.axs[2].plot(time,self.kmp_pos[1,:],color="green")
        self.axs[3].plot(time,self.kmp_pos[2,:],color="green")
        self.axs[4].plot(time,self.kmp_pos[3,:],color="green")
        self.axs[1].plot(time,self.mu[0,:],linestyle="dashed",color="grey")
        self.axs[2].plot(time,self.mu[1,:],linestyle="dashed",color="grey")
        self.axs[3].plot(time,self.mu[2,:],linestyle="dashed",color="grey")
        self.axs[4].plot(time,self.mu[3,:],linestyle="dashed",color="grey")
        self.axs[0].scatter([p1[0,0],p2[0,0],p3[0,0],p4[0,0]],[p1[0,1],p2[0,1],p3[0,1],p4[0,1]])
        self.axs[1].scatter(t,[p1[0,0],p2[0,0],p3[0,0],p4[0,0]])
        self.axs[2].scatter(t,[p1[0,1],p2[0,1],p3[0,1],p4[0,1]])
        self.axs[3].scatter(t,[p1[0,2],p2[0,2],p3[0,2],p4[0,2]])
        self.axs[4].scatter(t,[p1[0,3],p2[0,3],p3[0,3],p4[0,3]])
        self.axs[0].set_xlim(-12,12)
        self.axs[0].set_ylim(-12,12)
        self.axs[1].set_xlim(0,time[-1])
        self.axs[1].set_ylim(-12,12)
        self.axs[2].set_xlim(0,time[-1])
        self.axs[2].set_ylim(-12,12)
        self.axs[3].set_xlim(0,time[-1])
        self.axs[3].set_ylim(-60,60)
        self.axs[4].set_xlim(0,time[-1])
        self.axs[4].set_ylim(-60,60)
        self.axs[0].set_xlabel(r"x [cm]")
        self.axs[0].set_ylabel(r"y [cm]")
        self.axs[1].set_xlabel(r"t [s]")
        self.axs[1].set_ylabel(r"x [cm]")
        self.axs[2].set_xlabel(r"t [s]")
        self.axs[2].set_ylabel(r"y [cm]")
        self.axs[3].set_xlabel(r"t [s]")
        self.axs[3].set_ylabel(r"$\dot x$ [cm/s]")
        self.axs[4].set_xlabel(r"t [s]")
        self.axs[4].set_ylabel(r"$\dot y$ [cm/s]")
        for ax in self.axs:
            ax.grid()
            ax.set_axisbelow(True)
        self.fig.suptitle('Learning and adaptation - linear velocity')
        self.fig.tight_layout()
        self.fig.show() 
