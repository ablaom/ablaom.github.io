---
layout: post
title:  "Improving parabolic extrapolation with a simple recursive neural network"
date: 2018-10-13 10:48:05 +1300
categories: [recursive neural networks, time series]
---

![]{{ base.url }}/assets/noisy100.png)


Naive quadratic extrapolation of time series (i.e., fitting a parabola
to the last three points of data) can be viewed as applying an
exceedingly simple [recursive neural
network](https://en.wikipedia.org/wiki/Recursive_neural_network) with
certain weights. Allowing the network to learn better weights from
concrete data, we obtain a superior extrapolation scheme with no added
complexity. We detail this idea here as a gentle introduction to
recursive neural network concepts. 

We suppose the reader is somewhat acquainted with conventional
feed-forward neural networks.

Julia code for the examples can be found my
[ParticleTracker](github.com/ablaom/ParticleTracker) GitHub repo.

## Quadratic extraploation

Suppose we are trying to predict the value $x_{n+1} $ in a 
sequence 

$x_0, x_1, x_2, \ldots$

using only the preceding values $x_0, x_1, \ldots, x_n$. In quadratic
extrapolation, we construct our guess, $y_n$ say, by fitting a
parabola $p(x)$ to the last three points in the graph, $P=(n-2,
x_{n-2})$, $Q=(n-1, x_{n-1})$, $R=(n, x_n)$, and setting $y_n = p(n + 1)$:

![]({{ base.url }}/assets/parabolic_extrapolation.jpg)

As is readily checkted, one obtains

$y_{n} = 3x_n - 3x_{n-1} + x_{n-2}.$

To shift to the recursive neural network viewpoint, we think of $x_0,
x_1, \ldots, x_n$ as "inputs" fed sequentially into a black box, which
sequentially outputs predictions $y_0, y_1, \ldots, y_n$ given by the
formula above. (To get the first three
predictions $y_0, y_1, y_2$ we must also specify "warm-up" inputs
$x_{-2}$ and $x_{-1}$). Internal to the black box are two "hidden"
states, which remember just enough of what went before to keep the
predictions coming. Specifically, defining $h_n=x_n$ and
$g_n=x_{n-1}$, we may rewrite the above equation as three coupled
equations,

$ y_{n} = 3x_n -3h_{n-1} + g_{n-1}, \\
h_n = x_n, \\
g_n = h_{n-1}.$

Written this way, the output (prediction) $y_n$ depends only on the
current input $x_n$ and the *immediately preceding values* of the
hidden variables, while updates to the hidden variables depend only on
their immediately preceding values and the current input $x_n$. *Every
recursive neural network is built out of fundamental elements (called
cells) with this fundamental property.* 

In the present case we have one-dimensional inputs, one-dimensional
outputs, and a two-dimensional hidden state, but in general a
recurrent neural network might have inputs, outputs, and hidden states
of any dimension.

Instead of sequentially feeding inputs into a single black box, we may
equivalently imagine feeding all inputs simultaneously into an array
of serially connected identical copies of our cell, as shown below:

![]({{ base.url }}/assets/unrolled.jpg)

In this way we obtain the feed-forward neural network, known as an
*unrolling* of our recursive network. Fixing $n$, we may understand
this to be a network with $n$ inputs and $n$ outputs (not counting the
warm-up inputs; see more on this below). Notice that an input signal
at $x_0 has a long way to travel to reach the output $y_n$, making the
network effectively $n$-layers deep.


## Learning new weights

Let's see how good a job quadratic interpolation performs on some
concrete data, a discrete non-linear oscillator (see the appendix
below). In blue is part of the series and in red the predictions of
quadratic interpolation based on the preceding three states of the
oscillator:

![]({{ base.url }}/assets/smooth/ParabolicvsTruthCurves.png)

Actually, you can hardly detect a difference. However, if we introduce
noise to the oscillator, we get quite a different outcome:

![]({{ base.url }}}/assets/noisy/ParabolicvsTruthCurves.png)

To improve our extrapolator, we frist replace equations defining our
recursive neural network with equations having the same form but more
general coefficients:

$ y_{n} =  A x_n + B h_{n-1} + C g_{n-1}, \\
h_n =  D x_n + E h_{n-1} + F g_{n-1}, \\
g_n = J x_n + K h_{n-1} + L g_{n-1}.$

Note that the output $y_n$ continues to depend only on the current
input $x_n$ and the immediately preceding hidden states (of which
there are still only two) while the new hidden states depend their
immediately preceeding values and the current input $x_n$. (In the
GitHub code the variables have different names: `u = A, V = [D, J], Wy
= [B C], Wh = [E F; K L]`; and we fix `D=1`, as it turns out there is
no loss of generality in doing so.) 

Our next step is to choose the network weights (values for the
coefficients $A, B, C, D, E, F, J , K, L $) that will optimize an
appropriate training loss function. To these weights to be optimized
we add the warm-up inputs $h_{-1}$ and $g_{-1}$. In this simple
illustration, we can generate arbitrarily lengths of data, and
therefore dismiss concerns about overfitting the training data. We
also have some discretion as to what exactly constitutes "training",
and in the choice of loss function. 

In this case we train our network on sequences of length 100. Details
now follows (skip below to see the results!). We generate a noisy
oscillator sequence of length 10,000 and divide it into 100
consecutive subsequences $X_1, X_2, \ldots, X_{100}$ of length 100
each. If we feed the sequence $X_1$ into the 100-deep unrolled version
of our network, then the last output - let's call this 100th output
$\hat y_1$ - is our extrapolator's prediction for the *first* element
of subsequence $X_2$, which we will call $y_1$. Moving to the next
blocks we obtain $\hat y_2$ and $y_2$, and so on. We choose as
training loss the root-mean-square of the deviations $\hat y_1 - y_1$,
$\hat y_2 - y_2$, $\ldots$, $\hat y_{100} - y_{100}$, which we
minimize using usual stochastic gradient descent. Our code uses an
automatic differentiation library to compute gradients but one can use
traditional back-propagation as well. One does need to keep in mind
that our unrolled network has a lot of shared weights.

To formally compare the performance of our two extrapolators, we
simply fed a new 10,0000 long noisy oscillator sequence through our
recurrent network, from beginning to end, and computed the
root-mean-square of deviations between the original input sequence,
delayed by one time step, with the output sequence. However, a sample
plot (for the same data used above) will suffice to convince you of
the superior performance of the optimised recursive neural network:

![]({{ base.url }}assets/noisy/RNNvsTruthCurves.png)

	
## Appendix. Time series from non-linear oscillations of a particle

Let us briefly our method for generating a sequence of numbers of
*second order*, meaning the next value in the sequence depends only on
the current and previous two values. This discussion assumes some
basic physics and ordinary differential equations.

Consider a particle moving in one-dimension, under the influence of a
force field $F$. We suppose the particle has unit mass. We can obtain
a "discrete" version of the motion by applying a numerical technique
known as [leap frog
integration](https://en.wikipedia.org/wiki/Leapfrog_integration) to
the equations of motion. The accuracy of the integration is controlled
by a time-step parameter, $\Delta t$, which here we set to one. (There
is no loss in generality in doing so, for taking $\Delta t \neq 0$ is
the same as rescaling $F$ by $ 1/\Delta t^2$.) If the initial position
of the particle is $x_0$, and its initial velocity is $v_0$, then in
the discrete model the position after integer time $t$ is given by

$  x_{t+1} = x_t + v_t + \frac{1}{2} a_t\\
v_{t+1} = v_t + \frac{1}{2}(a_t + a_{t+1}),
$

where $a_t=F(x_t)$. Of course these equations can also be viewed as a
recurrent neural network in disguise, although more complicated than
the ones considered in the previous sections, because of the
non-linearity of $F$.

We constrain the particle to remain bewteen $- 1$ and $+ 1$ by
choosing 

$F(x) = -\frac{8\pi}{T^2}\tan(\frac{\pi x}{2})$. 

Here $T$ is a parameter
controlling the frequency of oscillations: if we linearize the
equations of motion about the stable solution $x = 0$, then the period
of the undiscretized motion is exactly $T$. The period of our
oscillator, will be then approximately $T$ steps. For $T=25$, a
typical sequence generated by the above equations is shown below (with
straight-line segments joining successive states):

![]({{ base.url }}/assets/smooth100.png)

We introduce noise of strength $\epsilon \ge 0$ as follows: Each
prediction $x_t$, calculated according to the above formulas, is
replaced with $(1-\epsilon)x_t + \epsilon \xi_t$, where $\xi_t$ is a
number drawn at random from the interval $[-1, 1]$ according to the
uniform distribution. Choosing $\epsilon = 0.01$, we obtain, for
example, the sequence illustrated below, drawn from the longer
sequence shown below it:

![]{{ base.url }}/assets/noisy100.png)

![]({{ base.url }}/assets/noisy.png)


