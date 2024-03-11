import time
import matplotlib.pyplot as plt
import numpy as np

import os

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from sklearn import datasets, svm, metrics

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import pdb

class NN_Helper():
    def __init__(self, **params):
        self.X, self.y = None, None
        return

    def sigmoid(self, x):
        x = 1/(1+np.exp(-x))
        return x

    def sigmoid_grad(self, x):
        return np.exp(-x) / (1 + np.exp(-x))**2
    

    def plot_activations(self, x):
        sigm   = self.sigmoid(x)
        d_sigm = self.sigmoid_grad(x)
        d_tanh = 1 - np.tanh(x)**2
        d_relu = np.zeros_like(x) +  (x >= 0)

        fig, axs = plt.subplots(3,2, figsize=(16, 8))
        _ = axs[0,0].plot(x, sigm)
        _ = axs[0,0].set_title("sigmoid")
        _ = axs[0,0].set_xlabel("$y_{(l-1)} \cdot W_{l,j}$", fontsize=14)
        
        _ = axs[0,1].plot(x, d_sigm)
        _ = axs[0,1].set_title("derivative sigmoid")
        _ = axs[0,1].set_xlabel("$y_{(l-1)} \cdot W_{l,j}$", fontsize=14)
        
        _ = axs[1,0].plot(x, np.tanh(x))
        _ = axs[1,0].set_title("tanh")
        _ = axs[1,0].set_xlabel("$y_{(l-1)} \cdot W_{l,j}$", fontsize=14)
        
        _ = axs[1,1].plot(x, d_tanh)
        _ = axs[1,1].set_title("derivative tanh")
        _ = axs[1,1].set_xlabel("$y_{(l-1)} \cdot W_{l,j}$", fontsize=14)
        
        _ = axs[2,0].plot(x, np.maximum(0.0, x))
        _ = axs[2,0].set_title("ReLU")
        _ = axs[2,0].set_xlabel("$y_{(l-1)} \cdot W_{l,j}$", fontsize=14)
        
        _ = axs[2,1].plot(x, d_relu)
        _ = axs[2,1].set_title("derivative ReLU")
        _ = axs[2,1].set_xlabel("$y_{(l-1)} \cdot W_{l,j}$", fontsize=14)

        _ = fig.tight_layout()
        return fig, axs

    
    def NN(self, W,b):
        """
        Create a "neuron" z = ReLu( W*x + b )
        Returns dict
        - key "x": range of input values x
        - key "y": y = W*x + b
        - Key "z": z = max(0, y)
        """
        x = np.linspace(-100, 100, 100)
        z = W*x + b
        
        y = np.maximum(0, z)
        return { "x":x,
                 "y":y,
                 "W":W,
                 "b":b
                 }


    def plot_steps(self, xypairs):
        fig, ax = plt.subplots(1,1, figsize=(10,6))
        for pair in xypairs:
            x, y, W, b = [ pair[l] for l in ["x", "y", "W", "b" ] ]
            _ = ax.plot(x, y, label="{w:d}x + {b:3.2f}".format(w=W, b=b))
            
            _ = ax.legend()
            _ = ax.set_xlabel("$y_{(l-1)} \cdot W_{l,j}$", fontsize=14)
            #_ = ax.set_xlabel("x")
            _ = ax.set_ylabel("activation")
            _ = ax.set_title("Binary Switch creation")

        _ = fig.tight_layout()
        return fig, ax

    def step_fn_plot(self, visible=True):
        slope = 1000
        start_offset = 0

        start_step = self.NN(slope, -start_offset)

        end_offset = start_offset + .0001

        end_step = self.NN(slope,- end_offset)

        step= {"x": start_step["x"], 
               "y": start_step["y"] - end_step["y"],
               "W": slope,
               "b": 0
              }
        fig, ax = self.plot_steps( [  step ] )

        if not visible:
            plt.close(fig)

        return fig, ax
            
    def sigmoid_fn_plot(self, visible=True):
        fig, ax = plt.subplots(1,1, figsize=(6,4))
        x =np.arange(-5,5, 0.1)
        sigm   = self.sigmoid(x)
        _ = ax.plot(x, sigm)
        _= ax.set_title("sigmoid")
        _= ax.set_xlabel("$y_{(l-1)} \cdot W_{l,j}$", fontsize=14)

        if not visible:
            plt.close(fig)

        return fig, ax
    def plot_loss_fns(self):
        # prod = y * s(x)
        # Postive means correctly classified; negative means incorrectly classified
        prod  = np.linspace(-1, +2, 100)

        # Error if product is negative
        error_acc  =  prod < 0
        error_exp  =  np.exp( -prod )

        # Error is 0 when product is exactly 1 (i.e., s(x) = y = 1)
        error_sq    =  (prod -1 )** 2

        # Error is negative of product
        # Error unless product greater than margin of 1
        error_hinge =  (- (prod -1) ) * (prod -1 < 0)

        fig, ax = plt.subplots(1,1, figsize=(10,6))
        _ = ax.plot(prod, error_acc, label="accuracy")
        _ = ax.plot(prod, error_hinge, label="hinge")
        
        # Truncate the plot to keep y-axis small and comparable across traces
        _ = ax.plot(prod[ prod > -0.5], error_exp[ prod > -0.5], label="exponential")
        
        _ = ax.plot(prod[ prod > -0.5], error_sq[ prod > -0.5], label="square")
        _ = ax.legend()
        _ = ax.set_xlabel("error")
        _ = ax.set_ylabel("loss")
        _ = ax.set_title("Loss functions")



    def plot_cosine_lr(self):
        num_batches= 1000
        epochs = np.linspace(0, num_batches, 100)/num_batches
        coss = np.cos( np.pi * epochs )
        rates = 0.5 * (1 + coss)

        fig, ax = plt.subplots(1,1, figsize=(10,4))
        _ = ax.plot(epochs, rates)
        _  = ax.set_xlabel("Epoch")
        _  = ax.set_ylabel("Fraction of original rate")
        _  = ax.set_title("Cosine Learning Rate schedule")

        return fig, ax

class Charts_Helper():
    def __init__(self, save_dir="/tmp", visible=True, **params):
        """
        Class to produce charts (pre-compute rather than build on the fly) to include in notebook

        Parameters
        ----------
        save_dir: String.  Directory in which charts are created
        visible: Boolean.  Create charts but do/don't display immediately
        """
        self.X, self.y = None, None
        self.save_dir = save_dir

        self.visible = visible

        nnh = NN_Helper()
        self.nnh = nnh

        return

    def create_activation_functions_chart(self):
        nnh = self.nnh
        visible = self.visible
        
        fig, axs = nnh.plot_activations( np.arange(-5,5, 0.1) )
        
        if not visible:
            plt.close(fig)

        return fig, axs

    def create_sequential_arch_chart(self, visible=None):
        if visible is None:
            visible = self.visible
        
        # Define rectangle properties
        rect_width = 0.5
        rect_height = 1.5  # Adjusted height for longer rectangles
        spacing = 1.2  # Adjusted spacing for longer rectangles

        # Create figure and axis
        fig, ax = plt.subplots()

        # Draw rectangles and arrows
        for i in range(5):
            rect = plt.Rectangle((i*spacing, 0), rect_width, rect_height, color='lightgrey', edgecolor='black')
            ax.add_patch(rect)

            if i < 4:
                ax.annotate('', xy=((i+1)*spacing, rect_height/2), xytext=(i*spacing + rect_width, rect_height/2),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='black'))

        # Set axis limits and labels
        ax.set_xlim(-0.5, 6)  # Adjusted limit for longer rectangles
        ax.set_ylim(0, 2)
        ax.set_aspect('equal')
        ax.axis('off')

        if not visible:
            plt.close(fig)

        return fig, ax

    def create_functional_arch_chart(self, visible=None):
        if visible is None:
            visible = self.visible

        # Define rectangle properties
        rect_width = 0.5
        rect_height = 1.5  # Adjusted height for longer rectangles
        spacing = 1.2  # Adjusted spacing for longer rectangles

        # Create figure and axis
        fig, ax = plt.subplots()

        # Draw rectangles and arrows
        for i in range(5):
            rect = plt.Rectangle((i*spacing, 0), rect_width, rect_height, color='lightgrey', edgecolor='black')
            ax.add_patch(rect)

            if i < 4:
                ax.annotate('', xy=((i+1)*spacing, rect_height/2), xytext=(i*spacing + rect_width, rect_height/2),
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='black'))

                if i == 1:
                    ax.annotate('', xy=(3*spacing + rect_width, rect_height), xytext=(1*spacing, rect_height),
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.5', color='black'))

        # Set axis limits and labels
        ax.set_xlim(-0.5, 6)  # Adjusted limit for longer rectangles
        ax.set_ylim(0, 2)
        ax.set_aspect('equal')
        ax.axis('off')

        if not visible:
            plt.close(fig)

        return fig, ax


    def draw_surface(self, visible=None):
        if visible is None:
            visible = self.visible


        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(111, projection='3d')

        # Create a 10x10 grid of points
        x = np.linspace(0, 10, 10)
        y = np.linspace(0, 10, 10)
        X, Y = np.meshgrid(x, y)

        # Define a simple quadratic function for Z
        #Z = np.sin(np.sqrt(X**2 + Y**2))

        Z = np.zeros_like(X) + 2
        Z += 0.1 * np.sin(.25*X)*np.cos(.25*Y)

        # Plot the surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis')


        # Add a color bar which maps values to colors
        fig.colorbar(surf, shrink=0.5, aspect=5)

        ax.set_zlim(1.75,2.25)

        # Label the axes
        ax.set_xlabel("$\mathbf{x}_1$", fontsize=18)
        ax.set_ylabel("$\mathbf{x}_2$", fontsize=18)
        ax.set_zlabel("$\mathbf{y}$", fontsize=18)
        
        if not visible:
            plt.close(fig)

        return fig, ax

    def add_shaded(self, ax, xmin, xmax, ymin, ymax):
        # Define the vertices of the shaded area polygon
        zmin = ax.get_zlim()[0]
        verts = [(xmin, ymin, zmin), (xmax, ymin, zmin), (xmax, ymax, zmin), (xmin, ymax, zmin)]

        # Create a Poly3DCollection and add it to the plot
        poly = Poly3DCollection([verts], alpha=0.5, facecolors='grey')
        ax.add_collection3d(poly)

      
    def create_charts(self):
        save_dir = self.save_dir

        print("Saving to directory: ", save_dir)
        
        print("Create Activation function chart")
        fig, ax = self.create_activation_functions_chart()
        act_func_file = os.path.join(save_dir, "activation_functions.png")
        fig.savefig(act_func_file)

        fig, ax = self.create_sequential_arch_chart()
        seq_arch_file =  os.path.join(save_dir, "tf_sequential_arch.png")
        fig.savefig(seq_arch_file)

        fig, ax = self.create_functional_arch_chart()
        func_arch_file =  os.path.join(save_dir, "tf_functional_arch.png")
        fig.savefig(func_arch_file)

        fig, ax = self.draw_surface()
        surface_chart_file_0 = os.path.join(save_dir, "surface_chart_0.png")
        fig.savefig(surface_chart_file_0)

        fig, ax = self.draw_surface()
        _= self.add_shaded(ax, 2, 8, 2, 3)
        surface_chart_file_1 = os.path.join(save_dir, "surface_chart_1.png")
        fig.savefig(surface_chart_file_1)

        fig, ax = self.draw_surface()
        _= self.add_shaded(ax, 8, 9, 2, 8)
        surface_chart_file_2 = os.path.join(save_dir, "surface_chart_2.png")
        fig.savefig(surface_chart_file_2)

        print("Done")
        
        return { "activation functions": act_func_file,
                 "TF Sequential arch" : seq_arch_file,
                 "TF Function arch"   : func_arch_file,
                 "surfaces": [ surface_chart_file_0, surface_chart_file_1, surface_chart_file_2 ]
                 }

