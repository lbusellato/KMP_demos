# Kernelized Movement Primitives

This repository contains a Python implementation of the Kernelized Movement Primitives 
paper by Huang et al. [[1](#references)].

# Usage

<img width="800" height="600" src="images/gui.png">

The code implements a GUI for ease of exploration of KMP's capabilities. When first launched, the GUI loads the demonstration database for the letter 'G' (top left plot), then initializes a Gaussian Mixture Model (GMM) on it, with 8 components (top center plot). The starting letter as well as the number of components can be changed by editing the corresponding text boxes in the bottom right and clicking the 'Update' button.

Then a reference trajectory (top right plot) is extracted from the GMM via Gaussian Mixture Regression (GMR), which is then used to initialize two istances of KMP (the two plots on the bottom left).

The two KMP plots are interactive, and new waypoints can be independently inserted by clicking the corresponding point on the plot while holding the CTRL key.

Finally, the superposition of multiple feasible trajectories (bottom right plot) is demonstrated by initializing another KMP instance that mixes the other two according to some priority values. The superposition updates as the other KMP instances are updated.

# References

#### [1] KMP: learning and adaptation 
[[Link to publication]](https://www.researchgate.net/publication/331481661_Non-parametric_Imitation_Learning_of_Robot_Motor_Skills)
```
@InProceedings{Huang19ICRA_1,
   Title = {Non-parametric Imitation Learning of Robot Motor Skills},
   Author = {Huang, Y. and Rozo, L. and Silv\'erio, J. and Caldwell, D. G.},
   Booktitle = {Proc. {IEEE} International Conference on Robotics and Automation ({ICRA})},
   Year = {2019},
   Address = {Montreal, Canada},
   Month = {May},
   Pages = {5266--5272}
}
```