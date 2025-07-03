# Fully Connected - Predictive Coding

Testing out an idea i got after watching [this](https://www.youtube.com/watch?v=l-OLgbdZ3kk) video on predictive coding. 
The idea essentially came from the thought "are layers even necessary here?"


## How it works

This revolves around a network $\mathcal{N}$ of nodes.

Each node has an activation, and an error value which is the difference between the activation and the _predicted_ activation.

Nodes are connected to one another with a _weight_ and an _activation function_

$$
\begin{align}
x_i &= \text{activation of node } i \\
\mu_i &= \text{predicted acativation of node } i  \equiv \sum_{j\in \mathcal{N}} w_{ji}f(x_j)\\
\varepsilon_i &= \text{error at node }i \equiv x_i-\mu_i
\end{align}
$$


The network is driven by minimizing a total energy
$$
E = \sum_i \varepsilon_i^2,
$$

### Some update rules

From principles of gradient descent

$$
\frac{\mathrm{d}x_i}{\mathrm{d}t} = -\gamma \frac{\partial E}{\partial x_i}
$$

$$
\frac{\mathrm{d}w_{ij}}{\mathrm{d}t} = -\gamma \frac{\partial E}{\partial w_{ij}}
$$


We can derive the following update rules for the weights and biases:

$$
x_i^{t+1} = x_i^t - \gamma \varepsilon_i^t +  \gamma f'(x_i^t)\sum_j \varepsilon_j^t w_{ij}^t 
$$

$$
w_{ij}^{t+1} = w_{ij}^t + \gamma \varepsilon_j^t f(x_i^t)
$$
