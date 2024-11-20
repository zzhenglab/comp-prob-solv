---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.13'
    jupytext_version: '1.16.4'
kernelspec:
  display_name: comp-prob-solv
  language: python
  name: python3
---

# Lecture 27: Kinetic Monte Carlo

## Learning Objectives

By the end of this lecture, you will be able to:

1. Understand the concept of Kinetic Monte Carlo (KMC) simulations.
2. Implement a simple KMC simulation in Python.
3. Apply KMC simulations to model the dynamics of a system.

## Introduction to Kinetic Monte Carlo

Phenomena such as diffusion, nucleation of a new phase, structural transformations, and chemical reactions often involve processes that occur over timescales much longer than those accessible via molecular dynamics (MD) simulations. MD simulations are limited by computational resources, making it challenging to simulate processes that occur over milliseconds or longer. Despite their extended timescales, these processes are kinetic and time-dependent.

A common aspect of these processes is that they proceed through transitions between various configurations or states, each occurring with a certain frequency. These frequencies determine the time evolution of the entire process. However, they are not outcomes of the kinetic Monte Carlo (KMC) method itself; rather, they must be known or calculated beforehand. This means that prior to performing a KMC simulation, the possible transitions involved in the kinetic process, along with their corresponding frequencies, must be identified and determined independently of the KMC calculation. These frequencies can be calculated using methods such as transition state theory or harmonic approximation. While they can also be calculated on the fly during the simulation, such calculations are separate from the KMC process itself.

KMC is a computational algorithm that simulates the time evolution of a system by probabilistically selecting and executing transitions based on their frequencies. To illustrate the KMC method in practice, we will apply it to model the diffusion process in a solid.

## Diffusion Studied by Kinetic Monte Carlo

We consider diffusion via the movement of vacancies in a crystal structure. For example, in the cubic structure of cerium dioxide (CeO₂), which crystallizes in this form, a vacancy can occur at one of the sites occupied by the majority atoms (oxygen in CeO₂). As shown in Figure 3, this vacancy can move to several neighboring sites, indicated by the arrows. Each such jump occurs with a specific frequency.

Vacancies facilitate atomic movement within the lattice, making them crucial for understanding diffusion mechanisms in solids. There are $N$ possible positions for the vacancy within the crystal lattice. If the vacancy is at position $i$, it can jump to $M_i$ different neighboring positions with frequencies $\nu_{ij}$, where $j$ indexes the possible destinations from position $i$. When it moves to a new position $j$, it can again jump to $M_j$ positions with frequencies $\nu_{jk}$, and so on. Thus, the diffusion process is characterized by the set of frequencies $\{ \nu_{ij} \}$, forming an $N \times N$ matrix where each element represents the frequency of a jump from position $i$ to position $j$. Note that some of these frequencies may be zero if a direct jump from $i$ to $j$ is not possible.

The probability that a jump from position $i$ to position $j$ occurs is given by:

$$
p_{ij} = \frac{\nu_{ij}}{\sum_{k=1}^{N} \nu_{ik}}
$$

Here, $\sum_{k=1}^{N} \nu_{ik}$ is the total frequency of all possible jumps from position $i$. Obviously, $0 \leq p_{ij} \leq 1$, and $\sum_{j=1}^{N} p_{ij} = 1$. These probabilities are commonly referred to as **transition probabilities**, while the frequencies $\nu_{ij}$ are known as **rate constants**.

The frequencies $\nu_{ij}$ can be calculated using methods such as transition state theory or derived from experimental data. In KMC simulations, these transition probabilities are used to stochastically determine the sequence of vacancy movements over time.

In defining these jump frequencies, we assume that when the vacancy moves from position $i$ to position $j$, it completely "loses the memory" of how it arrived at state $i$. Therefore, the probability $p_{ij}$ is entirely independent of the previous jumps that led to position $i$. This property characterizes the process as a **Markov process**, which is fundamental to the KMC method.

```{figure} ceo2-vacancy-diffusion.png
---
name: ceo2-vacancy-diffusion
---
Crystal structure of cerium dioxide (CeO₂) illustrating an oxygen vacancy (highlighted). The vacancy resides on an oxygen site within the fluorite structure. The arrows indicate potential pathways for oxygen ion diffusion via vacancy migration. Axes (a, b, c) are provided for reference.
```

## Kinetic Monte Carlo Procedure

We start with a vacanccy at a position marked $i$ and select randomly a possible jump into a position $j$. This jump is then made with probability $p_{ij}$. To do this we generate a random number $\xi$ such that $0 \leq \xi < 1$ and make the jump $i \rightarrow j$ if $p_{ij} \geq \xi$. However, if $p_{ij} < \xi$ we do not make any jump and repeat the process starting again by selecting randomly another possible jump.. After a jump occured we start with a new vacancy position that is associated with new jump frequencies (new rate constants) and the process is repeated. In this way the vacancy travels and attains various positions but to determine the rate at which it moves we have to associate a time with each step of this random walk.

If the vacancy is at a site $i$ then the frequency $\nu_\text{tot}^i$ with which it will leave the site $i$ is equal to the sum of the frequencies of all possible jumps away from the site $i$:

$$
\nu_\text{tot}^i = \sum_{k=1}^{M_i} \nu_{ik}
$$

Hence the time the vacancy stays at the site $i$ is

$$
t_i = \frac{1}{\nu_\text{tot}^i}
$$

In this way we have associated with every vacancy position $i$ the time $t_i$ during which the vacancy waits at this position. This time is usually dominated by one of the frequencies that is much higher than all the other frequencies. During the KMC process we associate with every state $i$ into which the vacancy got during the process, the time $t_i$ during which the vacancy remains at the position $i$, determined by the previous equation. The time associated with the process consisting of $K$ steps of the KMC is then

$$
\Delta t^{(K)} = \sum_{k \text{ corresponding to all states attained}} t_k
$$

## Transition (Reaction) Rate Theory

We investigate a process during which the potential energy of a system varies as depicted in the figure below. The initial and final states are **metastable**, corresponding to minima in the potential energy (or enthalpy, if external forces are performing work). Between these states, the energy rises to a maximum at an intermediate stage, representing the **activated state** or **transition state**.

```{code-cell} ipython3
:tags: [hide-input]
import matplotlib.pyplot as plt
import numpy as np

def add_double_arrow(x, y_min, y_max, arrow_length_offset=0, head_width=0.0125, head_length=0.05, color='black'):
    """
    Add a double-sided arrow to a plot.

    Parameters:
        x (float): The x-coordinate of the arrow base.
        y_min (float): The starting y-coordinate (bottom) of the arrow.
        y_max (float): The ending y-coordinate (top) of the arrow.
        arrow_length_offset (float): Adjustment to the arrow length (positive to make shorter).
        head_width (float): Width of the arrowhead.
        head_length (float): Length of the arrowhead.
        color (str): Color of the arrows.
    """
    arrow_length = y_max - y_min - arrow_length_offset
    
    # Downward arrow
    plt.arrow(x, y_min, 0, arrow_length, head_width=head_width, head_length=head_length, fc=color, ec=color)
    
    # Upward arrow
    plt.arrow(x, y_min + arrow_length, 0, -arrow_length + arrow_length_offset, head_width=head_width, head_length=head_length, fc=color, ec=color)

# Generate data for the potential energy curve
x = np.linspace(0, 1, 100)
y = -np.sin(2 * np.pi * x) ** 2 + x

# Define key points for the ground, activated, and final states
x_gs, x_as, x_fs = 0.25, 0.5, 0.75
y_gs = np.min(y)
y_as = np.max(y[y.shape[0] // 4:3 * y.shape[0] // 4])
y_fs = np.min(y[y.shape[0] // 2:])

# Plot the potential energy curve
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='Potential Energy Curve')
plt.xlabel('Reaction Coordinate')
plt.ylabel('Potential Energy')
plt.xlim(0.2, 0.8)
plt.xticks([x_gs, x_as, x_fs], ['Ground State', 'Activated State', 'Final State'])
plt.yticks([y_gs, y_as, y_fs], ['$E_\\text{gs}$', '$E_\\text{as} = E_\\text{gs} + E_a$', '$E_\\text{fs} = E_\\text{gs} + \\Delta E$'])

# Add dashed lines for energy levels
plt.axhline(y=y_gs, color='C1', linestyle='--', label='Ground State Level')
plt.axhline(y=y_fs, xmin=0.75, xmax=1, color='C2', linestyle='--', label='Final State Level')
plt.axhline(y=y_as, xmin=0.375, xmax=0.625, color='C3', linestyle='--', label='Activated State Level')

# Add double-sided arrows for activation energy and reaction energy
add_double_arrow(x=0.5, y_min=y_gs, y_max=y_as, arrow_length_offset=0.05)
add_double_arrow(x=0.75, y_min=y_gs, y_max=y_fs, arrow_length_offset=0.05)

# Add text annotations for energy levels
plt.text(0.5, (y_gs + y_as) / 2, '$E_a$', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='white'))
plt.text(0.75, (y_gs + y_fs) / 2, '$\\Delta E$', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='white'))

plt.legend()
plt.tight_layout()
plt.show()
```

As illustrated in the figure above, the potential energy profile shows two minima separated by an energy barrier of height $E_a$. The "reaction coordinate" axis represents the pathway along which the reaction occurs, encompassing the structural changes the system undergoes during the transformation. During this process, the system temporarily acquires an amount of energy equal to $E_a$ to overcome the transition state.

**We now pose the following question: Assuming we know the reaction path and the associated energy barrier of height $E_a$, what is the frequency with which the transition from one equilibrium state to the other occurs?**

This question can be answered to a good approximation within the framework of classical transition state theory (also known as reaction rate theory), which was first thoroughly presented in the book by Glasstone, Laidler, and Eyring:

[Glasstone, S., Laidler, K. J., and Eyring, H., *The Theory of Rate Processes*, McGraw-Hill, 1941](https://wash-primo.hosted.exlibrisgroup.com/permalink/f/1kqpcd6/WUSTL_SIERRA21314618).

Transition state theory is fundamental in chemical kinetics as it allows us to predict reaction rates based on the energy barrier and temperature. It provides a method for calculating reaction rates by considering the activated complex at the top of the energy barrier. In the following sections, we summarize the main aspects of this theory and derive an expression for the reaction rate constant $k$ that depends on $E_a$ and the temperature $T$.

## Basic Assumptions of Transition State Theory

1. **Stable Initial and Final States:**
   - The reaction involves initial and final configurations that are stable states. The transition between these states occurs along a specific pathway known as the **reaction coordinate** or **reaction path**.
   - The reaction coordinate represents a path on the **potential energy surface (PES)**, which maps the energy changes as the system progresses from reactants to products.

2. **Energy Barrier and Activated State:**
   - There is an energy barrier between the initial and final states when moving along the reaction path. The most favorable path is the one with the lowest energy barrier. The point at which the energy reaches a maximum along this path is called the **activated state** or **transition state**.
   - Although the transition state represents a maximum in energy, it is a transient configuration that cannot be isolated.

3. **Quasi-Equilibrium of the Activated State:**
   - It is assumed that the system reaches a **quasi-equilibrium** where the activated state is in thermodynamic equilibrium with the initial state. This allows for the use of equilibrium thermodynamics to describe the population of the activated state.
   - In the activated state, the system vibrates in all directions perpendicular to the reaction coordinate but moves translationally along the reaction path. In contrast, the stable initial and final states have vibrational degrees of freedom in all directions, including along the reaction coordinate.
   - The validity of these assumptions depends on the specific reaction mechanism. These are approximations, and experimental observations ultimately determine whether the conclusions drawn from the theory are accurate. In cases where quantum tunneling or non-equilibrium dynamics play a significant role, the assumptions may not hold.

## Main Results of Transition State Theory

In transition state theory, the reaction rate (or frequency of activations) is given by:

$$
\nu = \nu_0 \exp\left(-\frac{E_a}{k_\text{B} T}\right)
$$

where:

- $\nu$ is the reaction rate or frequency of successful transitions.
- $\nu_0$ is the **pre-exponential factor** or **attempt frequency**, representing the frequency of attempts to overcome the energy barrier.
- $E_a$ is the activation energy—the height of the energy barrier between the initial and final states.
- $k_\text{B}$ is the Boltzmann constant.
- $T$ is the absolute temperature.

Because of the exponential dependence on the activation energy $E_a$, the actual reaction frequency $\nu$ is much smaller than the attempt frequency $\nu_0$; the difference between these two frequencies can span several orders of magnitude.

```{tip}
To test whether a process is well described by transition state theory, you can produce an **Arrhenius plot**. Plot the logarithm of a quantity proportional to the reaction rate (e.g., $\ln \nu$ or $\ln k$) versus the inverse of the temperature $1/T$. If the process follows transition state theory, the plot should be a straight line, indicating an exponential dependence on $E_a$.
```
