# Kernelized Movement Primitives

This repository contains a Python implementation of the Kernelized Movement Primitives 
paper by Huang et al. [[1](#references)].

# Usage

Launch the main.py script with the number of the desired demo (i.e. 1 or 2) as argument.

## Demo 1

<img width="800" height="600" src="images/gui.png">

The demo implements a GUI for ease of exploration of KMP's capabilities. When first launched, the GUI loads the demonstration database for the letter 'G' (top left plot), then initializes a Gaussian Mixture Model (GMM) on it, with 8 components (top center plot). The starting letter as well as the number of components can be changed by editing the corresponding text boxes in the bottom right and clicking the 'Update' button.

Then a reference trajectory (top right plot) is extracted from the GMM via Gaussian Mixture Regression (GMR), which is then used to initialize two istances of KMP (the two plots on the bottom left).

The two KMP plots are interactive, and new waypoints can be independently inserted by clicking the corresponding point on the plot while holding the CTRL key.

Finally, the superposition of multiple feasible trajectories (bottom right plot) is demonstrated by initializing another KMP instance that mixes the other two according to some priority values. The superposition updates as the other KMP instances are updated.

## Demo 2

<img width="800" height="600" src="images/demo2.png">

The demo implements orientation learning with KMP [[2](#references)]. The code loads a set of demonstrations recorded on an UR5e manipulator and extracts the pose data (i.e. xyz coordinates and axis-angle representations of the orientations). For the positions, a KMP pipeline similar to that of demo 1 is implemented (top row). The orientations are converted to quaternions (middle row), which are then projected in Euclidean space (bottom row) and used to train a KMP model. The model is then used to show both learning, i.e. emulating the original reference trajectory, and adaptation, i.e. adapting towards a new desired quaternion.

# References

#### [1] KMP: learning and adaptation. 
[[Link to publication]](https://www.researchgate.net/publication/319349682_Kernelized_Movement_Primitives)
```
@Article{Huang19IJRR,
  Title = {Kernelized Movement Primitives},
  Author = {Huang, Y. and Rozo, L. and Silv\'erio, J. and Caldwell, D. G.},
  Journal = {International Journal of Robotics Research},
  Year = {2019},
  Volume={38},
  Number={7},
  Pages = {833--852},
}
```

#### [2] KMP: learning orientations.
[[Link to publication]](https://www.researchgate.net/publication/330675655_Generalized_Orientation_Learning_in_Robot_Task_Space)
```
@InProceedings{Huang19ICRA_2,
   Title = {Generalized Orientation Learning in Robot Task Space},
   Author = {Huang, Y. and Abu-Dakka, F. and Silv\'erio, J. and Caldwell, D. G.},
   Booktitle = {Proc. {IEEE} International Conference on Robotics and Automation ({ICRA})},　　　　
   Year = {2019},
   Address = {Montreal, Canada},
   Month = {May},
   Pages = {2531--2537}
 }
```