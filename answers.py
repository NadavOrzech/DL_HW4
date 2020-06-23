r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""


# ==============
# Part 1 answers


def part1_pg_hyperparams():
    hp = dict(batch_size=32,
              gamma=0.99,
              beta=0.5,
              learn_rate=1e-3,
              eps=1e-8,
              )
    # TODO: Tweak the hyperparameters if needed.
    #  You can also add new ones if you need them for your model's __init__.
    # ====== YOUR CODE: ======
    #hp['beta'] = 0.2
    # ========================
    return hp


def part1_aac_hyperparams():
    hp = dict(batch_size=32,
              gamma=0.99,
              beta=1.,
              delta=1.,
              learn_rate=1e-3,
              eps=1e-8,
              )
    # TODO: Tweak the hyperparameters. You can also add new ones if you need
    #   them for your model implementation.
    # ====== YOUR CODE: ======


    # ========================
    return hp


part1_q1 = r"""
**Your answer:
The policy gradient technic produces high variance, which as explained in the exercise causes erratic optimization
behavior and slow convergence. This happens because when we compute the gradient we sample trajectories from the current
policy and average their values thus making the computed gradient dependant on the randomly sampled trajectories. 
Subtracting an appropriate baseline we will cause the expectation to be clustered closer to zero, which will cause the
variance to drop.

For example in our game, lets say that the actions probabilities are [0.4, 0.1, 0.3, 0.2] and the reward for the four 
trajectories are [1000, 1001, 1003, 1010]. The variance for this parameters is 2.9 * 10e8. If we will choose baseline of
1003 we will calculate variance for {0.4*-3, 0.1*-2, 0.5*0, 0.2*7} which value is 0.413. As we explained the variance 
dropped dramatically and as a result improves convergence.
**



Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part1_q2 = r"""
In AAC, when using the estimated q-values as regression targets for our state-values, why do we get a valid 
approximation? Hint: how is  𝑣𝜋(𝑠)  expressed in terms of  𝑞𝜋(𝑠,𝑎) ?
**Your answer:

**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part1_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
