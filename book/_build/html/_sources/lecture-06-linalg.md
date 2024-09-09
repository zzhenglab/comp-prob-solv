---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: comp-prob-solv
  language: python
  name: python3
---

# Lecture 6: Balancing Chemical Equations and Systems of Linear Algebraic Equations

### Balancing Chemical Equations

One of the first things we're trained to do in the chemical sciences is to balance the stoichiometric coefficients of the reactants and products in a chemical equation so that no mass is lost. For example, consider the combustion of an methane in the presence of oxygen.

$$
\begin{align*}
a \text{CH}_4(g) + b \text{O}_2(g) &\rightarrow c \text{CO}_2(g) + d \text{H}_2\text{O}(g) \\
\end{align*}
$$

The game that we are taught to play is to determine the values of $a$, $b$, $c$, and $d$ that make the number of atoms of each element the same on both sides of the equation. To do this, we can make a table where the two columns are for reactants (left) and products (right) and each row is for a different element. For the combustion of methane, the table looks like this:

| Element | Reactants | Products |
|---------|-----------|----------|
| C       | $a$       | $c$      |
| H       | $4a$      | $2d$     |
| O       | $2b$      | $2c + d$ |

From the table, we can clearly see that $a$ has to be equal to $c$ in order for the reaction to be balanced. So, we can write the table as:

| Element | Reactants | Products |
|---------|-----------|----------|
| C       | $a$       | $a$      |
| H       | $4a$      | $2d$     |
| O       | $2b$      | $2a + d$ |

OK, the next step is to notice that $4a = 2d$ so $d = 2a$. We can substitute this into the last row of the table to get:

| Element | Reactants | Products |
|---------|-----------|----------|
| C       | $a$       | $a$      |
| H       | $4a$      | $4a$     |
| O       | $2b$      | $2a + 2a$ |

Finally, we can see that $2b = 4a$ so $b = 2a$. Substituting this into the last row of the table gives:

| Element | Reactants | Products |
|---------|-----------|----------|
| C       | $a$       | $a$      |
| H       | $4a$      | $4a$     |
| O       | $4a$      | $4a$     |

So, the balanced equation is:

$$
\begin{align*}
\text{CH}_4(g) + 2\text{O}_2(g) &\rightarrow \text{CO}_2(g) + 2\text{H}_2\text{O}(g) \\
\end{align*}
$$

Maybe that was unnecessarily difficult, as I'm sure many of you were able to see precisely what those values were before we started. But, the point is that we can solve linear equations to balance chemical equations.

### Systems of Linear Algebraic Equations

Going back even further, you may remember that if we have a system of linear algegraic equations and the same number of equations as unknowns, we can solve for the unknowns. Let's recast our combustion problem in terms of a system of linear algebraic equations. We have:

$$
\begin{align*}
a - c &= 0 \\
4a - 2d &= 0 \\
2b - 2c - d &= 0 \\
\end{align*}
$$

This can be written in matrix form as:

$$
\begin{align*}
\begin{bmatrix}
1 & 0 & -1 & 0 \\
4 & 0 & 0 & -2 \\
0 & 2 & -2 & -1 \\
\end{bmatrix}
\begin{bmatrix}
a \\
b \\
c \\
d \\
\end{bmatrix}
&=
\begin{bmatrix}
0 \\
0 \\
0 \\
\end{bmatrix}
\end{align*}
$$

Or, more simply as:

$$
\begin{align*}
\mathbf{A}\mathbf{x} &= \mathbf{0}
\end{align*}
$$

Where $\mathbf{A}$ is the matrix of coefficients, $\mathbf{x}$ is the vector of unknowns, and $\mathbf{0}$ is the zero vector. We can solve this system of equations by finding the null space of $\mathbf{A}$. The null space is the set of all vectors that send $\mathbf{A}$ to $\mathbf{0}$. In other words, it is the set of all vectors that satisfy the system of equations.

### Solving the System of Equations

Let's solve the system of equations for the combustion of methane using Python. We'll use the `numpy` library to do this.

First, we need to import the `numpy` library.

```{code-cell} ipython3
import numpy as np
```

Next, we'll define the matrix of coefficients, $\mathbf{A}$.

```{code-cell} ipython3
A = np.array([[1, 0, -1, 0],
              [4, 0, 0, -2],
              [0, 2, -2, -1]])
```

Now, we can find the null space of $\mathbf{A}$.

```{code-cell} ipython3
from scipy.linalg import null_space

# Calculate the null space of matrix A
null_space = null_space(A)
```

Finally, we can print the solution.

```{code-cell} ipython3
# Convert the null space solution to integer coefficients by multiplying
# and normalizing to the smallest integers
coefficients = null_space[:, 0]
coefficients = coefficients / np.min(coefficients[coefficients > 0])
coefficients = np.round(coefficients).astype(int)

coefficients
```

This is the same solution we found earlier. The balanced equation is:

$$
\begin{align*}
\text{CH}_4(g) + 2\text{O}_2(g) &\rightarrow \text{CO}_2(g) + 2\text{H}_2\text{O}(g) \\
\end{align*}
$$

```{admonition} Note
:class: tip
There's actually a general equation for the stoichiometric coefficients for the combustion of alkanes. For the combustion of an alkane with $n$ carbons, the balanced equation is:

$$
\begin{align*}
\text{C}_n\text{H}_{2n+2}(g) + (3n+1)\text{O}_2(g) &\rightarrow n\text{CO}_2(g) + (n+1)\text{H}_2\text{O}(g) \\
\end{align*}
$$

There's also a general function for any hydrocarbon, saturated or unsaturated. It is:

$$
\begin{align*}
\text{C}_x\text{H}_y + \left(x + \frac{y}{4}\right)\text{O}_2 &\rightarrow x\text{CO}_2 + \frac{y}{2}\text{H}_2\text{O} \\
\end{align*}
$$
```

### Example: Reduction of Tin(IV) Oxide by Hydrogen

Let's consider the reduction of tin(IV) oxide by hydrogen to form tin and water.

$$
\begin{align*}
\text{SnO}_2(s) + \text{H}_2(g) &\rightarrow \text{Sn}(s) + \text{H}_2\text{O}(g) \\
\end{align*}
$$

```{admonition} Wait a Minute!
:class: warning
Can you try to do this by hand before you do it in Python?
```

```{code-cell} ipython3

```
