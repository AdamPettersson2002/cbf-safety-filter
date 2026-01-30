# CBF-Based Safe Quadrotor Navigation

A Python implementation of Control Barrier Function (CBF) based safety-critical control for 3D quadrotor navigation with no-fly zone avoidance.

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

This project implements a safety filter using Control Barrier Functions (CBFs) to ensure a quadrotor safely navigates to a goal position while avoiding no-fly zones. The system uses:

- **Double integrator dynamics** for simplified 3D motion modeling
- **Exponential CBF (ECBF)** constraints for relative degree 2 systems
- **Quadratic Programming (QP)** to optimally modify unsafe control inputs
- **Modified Proportional Navigation Control** as the nominal goal-reaching policy


## Quick Start

### 1. Installation

**Recommended:** Use **Python 3.12** or **3.13** to avoid C++ compiler requirements for the QP solver.

```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/cbf-drone-sim.git](https://github.com/YOUR_USERNAME/cbf-drone-sim.git)
cd cbf-drone-sim

# Create a virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Activate (Mac/Linux)
source .venv/bin/activate

# Install dependencies
pip install numpy scipy matplotlib osqp 
```

## Usage & Scenarios
The project uses a scenarios.py file to define different test cases. You can switch scenarios by changing the string in simulation.py:

```python
if __name__ == "__main__":
    run_simulation("head_on", animate=True)
    #run_simulation("head_on_2", animate=True)
    #run_simulation("chase", animate=True)
    #run_simulation("many_blockers", animate=True)
    #run_simulation("los_problem", animate=True)
```

## Project Structure
The code is organized to separate concerns (Guidance vs. Safety vs. Physics).
```
cbf-safety-filter/
├── sim.ipynb           # MAIN ENTRY POINT: Loop, visualization, and logic
├── scenarios.py        # DEFINITIONS: Target paths, obstacle positions
├── guidance.py         # THE BRAIN: Target Tracking (u_nom)
├── safety_filter.py    # SAFETY FILTER: CBF-QP Solver (u_safe)
├── constraints.py      # MATH: Barrier Function definitions (h, h_dot)
├── dynamics.py         # PHYSICS: 3D Double Integrator
└── README.md
```
## Mathematical Background

### 1. System Dynamics (Double Integrator)

The state is defined as $x = [p, v]^T \in \mathbb{R}^6$.

$$
\dot{p} = v
$$
$$
\dot{v} = u
$$

### 2. Nominal Guidance (PPN)

We use Proportional Navigation to generate an acceleration perpendicular to the LOS vector of the target,
ffectively putting the drone on a collision course. Given the relative position

$$
\mathbf{r} = \mathbf{p}_{target} - \mathbf{p}_{drone},
$$

and relative velocity

$$
\mathbf{v}_{rel} = \mathbf{v}_{target} - \mathbf{v}_{drone},
$$

the rotation rate of the LOS vector is given by

$$
\boldsymbol{\Omega} = \frac{\mathbf{r} \times \mathbf{v}_{rel}}{\|\mathbf{r}\|^2}.
$$

The nominal acceleration input is then composed of the PN term and a push term parallell to the LOS vector:

$$
\mathbf{u}_{nom} = \underbrace{N (\boldsymbol{\Omega} \times \mathbf{v}_{closing})}_{\text{PN Guidance}} + \underbrace{k_p \frac{\mathbf{r}}{\|\mathbf{r}\|}}_{\text{Approach Push}} + \boldsymbol{\eta},
$$

where $N$ is the navigation gain (set to 4.0).

$\mathbf{v}_{closing} = -\mathbf{v}_{rel}$ is the closing velocity vector.

$k_p$ is a proportional gain (set to 2.0) to encourage movement toward the target.

$\boldsymbol{\eta} \sim \mathcal{N}(0, \sigma^2)$ represents added process noise.


### 3. Safety Filter (CBF-QP)

We minimize the deviation from the nominal control subject to the safety constraint.

**Optimization Problem:**

$$
\min_{u} \quad \frac{1}{2} ||u - u_{nom}||^2
$$

**Subject to:**

$$
A_{cbf} u \leq b_{cbf}
$$
$$
u_{min} \leq u \leq u_{max}
$$

**Barrier Condition:**
For relative degree 2 systems (acceleration controlled), the safety constraint is defined as:

$$
\ddot{h} \geq -k_1 \dot{h} - k_0 h
$$

This ensures the drone never enters the obstacle region defined by $h(x) < 0$.

## Troubleshooting

**QP solver failed**
- Increase safety margin `d_safe`
- Reduce CBF aggressiveness (`k0`, `k1`)
- Check that obstacles don't block all paths to goal

**Vehicle oscillates near obstacles**
- Reduce controller gains `kp`, `kd`
- Increase CBF damping parameter `k1`

**Slow convergence to goal**
- Increase controller gains `kp`, `kd`
- Reduce safety margin if too conservative

## Future Enhancements

- [ ] Multi-agent coordination
- [ ] Sensor modelling and FOV restrictions
- [ ] EKF Implementation
- [ ] C++ QP solver for faster performance

## License

This project is licensed under the MIT License - see the LICENSE file for details.
