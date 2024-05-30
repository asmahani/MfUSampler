# MfUSampler
Multivariate-from-Univariate (MfU) Markov Chain Monte Carlo Sampler provides machinery for generating samples from multivariate probability distributions using univariate sampling algorithms such as slice sampler and adaptive rejection sampler. The multivariate wrapper
performs a full cycle of univariate sampling steps, one coordinate at a time. In each step, the latest sample values obtained for other coordinates are used to form the conditional distributions. The concept is an extension of Gibbs sampling where each step involves,
not an independent sample from the conditional distribution, but a Markov transition for which the conditional distribution is invariant. The software relies on proportionality of conditional distributions to the joint distribution to implement a thin wrapper for producing conditionals. See also:
- [`MfUSampler` R package on CRAN](https://cran.r-project.org/web/packages/MfUSampler/index.html)
- [Paper published in *Journal of Statistical Software* describing the `MfUSampler` methodology as well as performance improvement strategies](https://www.jstatsoft.org/article/view/v078c01)
