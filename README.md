# CBF-Based Safe Quadrotor Navigation

A Python implementation of Control Barrier Function (CBF) based safety-critical control for 3D quadrotor navigation with no-fly zone avoidance.

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

This project implements a **safety filter** using Control Barrier Functions (CBFs) to ensure a quadrotor safely navigates to a goal position while avoiding no-fly zones. The system uses:

- **Double integrator dynamics** for simplified 3D motion modeling
- **Exponential CBF (ECBF)** constraints for relative degree 2 systems
- **Quadratic Programming (QP)** to optimally modify unsafe control inputs
- **Control Lyapunov Function (CLF) controller** as the nominal goal-reaching policy

### Key Features

✅ **Mathematically rigorous safety guarantees** via CBF theory  
✅ **Real-time QP-based safety filtering** using OSQP  
✅ **Multiple no-fly zone support**  
✅ **Comprehensive visualization** of trajectory, safety margins, and control signals  
✅ **Clean, modular code structure** with detailed documentation  

## 🚀 Quick Start

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

## 🕹 Usage & Scenarios
The project uses a scenarios.py file to define different test cases. You can switch scenarios by changing the string in simulation.py:

```python
if __name__ == "__main__":
    # Options: "head_on", "chase", "clutter"
    run_simulation("chase", animate=True)
```

## 📂 Project Structure
The code is organized to separate concerns (Guidance vs. Safety vs. Physics).
```
cbf-drone-sim/
├── simulation.py       # MAIN ENTRY POINT: Loop, visualization, and logic
├── scenarios.py        # DEFINITIONS: Target paths, obstacle positions
├── guidance.py         # THE BRAIN: LQR Target Tracking (u_nom)
├── safety_filter.py    # THE GUARDIAN: CBF-QP Solver (u_safe)
├── constraints.py      # MATH: Barrier Function definitions (h, h_dot)
├── dynamics.py         # PHYSICS: 3D Double Integrator
└── README.md
```
## 📐 Mathematical Background

### 1. System dynamics
The state is defined as $x = [p, v]^T \in \mathbb{R}^6$.
$$
\begin{aligned}
\dot{p} = v \\ \dot{v} = u
\end{aligned}
$$

### 2. Nominal Guidance (LQR)
We solve the Algebraic Riccati Equation (ARE) to find the optimal gain matrix $K$ that minimizes the error to the 
target:
$$
u_{nom} = -K (x_{drone} - x_{target})
$$
### 3.Safety Filter (CBF-QP)
We minimize the deviation from the nominal control subject to the safety constraint:
$$
\begin{aligned}
\min_{u} \quad & \frac{1}{2} ||u - u_{nom}||^2 \\
\text{s.t.} \quad & \dot{h}(x, u) \geq -\gamma(h(x)) \\
& u_{min} \leq u \leq u_{max}
\end{aligned}
$$
Where the barrier condition $\dot{h} \geq -k_1 \dot{h} - k_0 h$ ensures the drone never enters the 
obstacle region defined by $h(x) < 0$.

## 🔧 Troubleshooting

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
- [ ] Model Predictive Control (MPC) integration
- [ ] ROS interface for real quadrotor testing
- [ ] C++ QP solver for faster performance

## References

1. Ames, A. D., et al. (2014). "Control Barrier Functions: Theory and Applications." *European Control Conference*.
2. Xiao, W., & Belta, C. (2021). "Control Barrier Functions for Systems with High Relative Degree." *IEEE CDC*.
3. Stellato, B., et al. (2020). "OSQP: An Operator Splitting Solver for Quadratic Programs." *Mathematical Programming Computation*.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
