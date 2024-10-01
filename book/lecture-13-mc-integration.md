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

# Lecture 13: Monte Carlo Integration

In Lecture 5, we explored numerical integration methods such as Riemann sums and the trapezoidal rule, which are **deterministic** approaches. These methods subdivide the integration range into small intervals, using structured grids to approximate the area under the curve. However, as the dimensionality of the problem increases, these methods become computationally expensive due to the "curse of dimensionality."

Today, we will shift our focus to **Monte Carlo integration**, a **stochastic** method for numerical integration that is particularly powerful for high-dimensional integrals and problems where deterministic methods are inefficient.

## Review: Deterministic Numerical Integration

Previously, we computed integrals using structured approaches:

1. **Riemann Sum:** We divided the interval of integration into small rectangles and approximated the area under the curve. The accuracy increased as the number of subdivisions grew.

2. **Trapezoidal Rule:** This method improved accuracy by approximating the function with straight-line segments instead of rectangles. We used the trapezoidal rule to compute both 1D and multidimensional integrals, such as the overlap integral of hydrogen 1s orbitals.

3. **Scaling Issues in High Dimensions:** For both methods, the number of function evaluations grows exponentially with the number of dimensions (i.e., $\mathcal{O}(n^d)$ for a $d$-dimensional integral). This makes these methods impractical for problems in quantum chemistry and statistical mechanics, where we may need to integrate over many dimensions.

## Key Advantages of Monte Carlo Integration

1. **Dimensional Independence**: Unlike deterministic methods, the error in Monte Carlo integration does not scale exponentially with the number of dimensions. Instead, the error scales as $\mathcal{O}(1/\sqrt{N})$, where $N$ is the number of random samples, independent of the number of dimensions.

2. **Handling Complex Domains**: Monte Carlo integration can handle integrals over complex or irregular domains, where structured grids might be inefficient or infeasible.
