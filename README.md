# Fully Connected - Predictive Coding

Testing out an idea i got after watching [this](https://www.youtube.com/watch?v=l-OLgbdZ3kk) video on predictive coding. 
The idea essentially came from the thought "are layers even necessary here?"


## Some update rules

From these principles

$$
\frac{\mathrm{d}}{\mathrm{d}x_i} = -\gamma \frac{\partial E}{\partial x_i}
$$

$$
\frac{\mathrm{d}}{\mathrm{d}w_{ij}} = -\gamma \frac{\partial E}{\partial w_{ij}}
$$

with 

$$
E = \sum_i \varepsilon_i^2,
$$
where 
$$
\varepsilon_i = x_i - \mu_i = x_i - \sum_j w_{ij}f(x_j)
$$


We can derive the following update rules for the weights and biases:

$$
x_i^{t+1} = x_i^t - \gamma \varepsilon_i^t +  \gamma \sum_j w_{ij}^t f'(x_i^t)
$$

$$
w_{ij}^{t+1} = w_{ij}^t + \gamma \varepsilon_j^t f(x_i^t)
$$
