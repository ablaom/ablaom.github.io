---
layout: post
title:  "Univariate interpolation: moving beyond moving averages"
date: 2017-10-09 11:36:50 +1300
categories: [regression]
---

![icecream data]({{base.url}}/assets/icecream.png) ![babel image]({{base.url}}/assets/babel.png)

[Here](https://rawgit.com/ablaom/Smoothers/master/smoothers.pdf) is an
article I wrote a little while ago describing a clever way to fuse
moving averages and simple linear regression to obtain a neat way of
smoothly interpolating univariate data (see the left figure). The method is known as
*locally linear regression*, which is a special case of
[kernel smoothing](https://en.wikipedia.org/wiki/Kernel_smoother).

The method can be generalized to interpolate multivariate data, but
the extra computational expense is usually prohibitive. In those
cases I would use locally constant regression (as discussed in the article) instead.
