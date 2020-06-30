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
    hp['beta'] = 0.1
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
    hp['gamma'] = 0.97
    hp['delta'] = 0.5
    hp['beta'] = 0.75
    hp['learn_rate'] = 3e-3

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




"""


part1_q2 = r"""
**Your answer:
As we learned the advantage function is the difference between the action-value function and the state-values function.
The state-value represents the value of the state by averaging over all the actions possible according to the policy. 
The action-value represents the value of the state according to a given the first action.
So we can see that the state-values are an average of the action-values of the same state, so by using the estimated 
q-values as regression targets we can estimate the average action-values thus calculate the state-values.
**


"""


part1_q3 = r"""
Analyze and explain the graphs you got in first experiment run.
Compare the experiment graphs you got with the AAC method to the regular PG method (cpg).
**Your answer:
1. In the first experiment run we produce four graphs from four different runs.

The first graph showcases the loss_p of the runs. For the two runs with a baseline the loss_p is the 
BaselinePolicyGradientLoss and for the two runs without a baseline the loss_p is the VanillaPolicyGradientLoss. We can 
see that for the runs without a baseline the loss_p is gradually ascending, meaning approaching 0, as we expect from a 
model that is learning correctly. For the baseline policy loss we see that the loss_p remains near 0, like we expect 
from it. As explained earlier in the exercise the policy gradient with baseline reduces the variance of our gradient 
using relative weighting of the log policy instead of absolute reward values.

The second graph showcases the loss_e of the run, i.e the entropy loss. Because only two runs are with entropy we get 
only two results. First, we can see that the results are in range of [-0.1, 0], this is because we normalize the entropy.
Also we see that the values are gradually ascending for both runs as we would expect because we aim to maximize the 
entropy loss throughout the run.

The third graph showcases the baseline of the two runs that are done with a baseline. For both of the runs as we 
progress the baseline improves and becomes bigger. This behavior is expected, because as we continue to train we expect
the reward to get bigger and the baseline is the average of the total rewards. 

The fourth graph showcases the mean_reward of all the runs. We can see that the main difference between the runs is that
adding a baseline improves the results. The two runs with a baseline produced better results than the runs without a 
baseline. For the two runs without a baseline the run with the entropy loss is slightly better than without, this can 
be explained because the entropy prevents the policy distribution from becoming too narrow and helps promote the agents
exploration, thus achieving better results.

2. 
Regarding the second graph of the loss_e, it's hard to compare the cpg and aac because there starting point is different.
The cpg start with beta = 0.1 while the aac starts with beta = 0.75. However they are both gradually ascending.

In the fourth graph, showcasing the mean_reward the results are similar, with the aac achieving a slightly higher reward.
For the aac instead of using the baseline we use in the cpg, we use the advantage function which is a smarter approach, 
and as expected it receives a higher reward. According to the actor-critic method we have to identical models that each 
train and improve thus improving also the other model, and make the advantage function a more powerful baseline.

As we explained the AAC method gives us the best results, as we can see as well on the loss_p graph. In the AAC case, 
the loss is calculated with a learnable baseline, which improves its value significantly. 

 
**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
