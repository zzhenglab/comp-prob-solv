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

# Lecture 15: Nanoparticle Shape and Simulated Annealing

## Learning Objectives

By the end of this lecture, you should be able to

1. Describe the role of shape in nanoparticle properties.
2. Explain how simulated annealing can be used to find the optimal shape of a nanoparticle.

## Nanoparticle Shape

The shape of a nanoparticle can have a significant impact on its properties. For example, the shape of a nanoparticle can affect its:

- Optical properties, by subjecting the nanoparticle to nanoscale boundary conditions.
- Mechanical properties, by truncating long-range interactions.
- Chemical properties, by changing the number and configuration of surface atoms.

## Local *vs.* Global Geometry Optimization

The shape of a nanoparticle can be optimized using a variety of methods. One common approach is to use a local optimization algorithm, such as those implemented in `scipy.optimize`. However, local optimization algorithms can get stuck in local minima, especially when the objective function is non-convex.

![Local optimization](local_minimum.png)


