import numpy as np
from scipy import integrate

# -------------------------------------------------
# Question 1 – Exponential Distribution
# -------------------------------------------------

def exponential_pdf(x, lam=1):
    """
    Return PDF of exponential distribution.
    f(x) = lam * exp(-lam*x) for x >= 0
    """
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, lam * np.exp(-lam * x), 0.0)

def exponential_interval_probability(a, b, lam=1):
    """
    Compute P(a < X < b) using analytical formula.
    """
    # P(a < X < b) = F(b) - F(a) = e^(-lam*a) - e^(-lam*b)
    return np.exp(-lam * a) - np.exp(-lam * b)

def simulate_exponential_probability(a, b, n=100000):
    """
    Simulate exponential samples and estimate
    P(a < X < b).
    """
    # numpy uses scale = 1/lam, so scale=1 means lam=1
    samples = np.random.exponential(scale=1.0, size=n)
    return np.mean((samples > a) & (samples < b))

# -------------------------------------------------
# Question 2 – Bayesian Classification
# -------------------------------------------------

def gaussian_pdf(x, mu, sigma):
    """
    Return Gaussian PDF.
    """
    coefficient = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coefficient * np.exp(exponent)

def posterior_probability(time):
    """
    Compute P(B | X = time)
    using Bayes rule.
    Priors:
    P(A)=0.3
    P(B)=0.7
    Distributions:
    A ~ N(40,4)
    B ~ N(45,4)
    """
    # Note: sigma=4 means std dev = 4 (NOT variance=4)
    # Check your course notes — if variance=4, use sigma=2
    prior_A, prior_B = 0.3, 0.7
    mu_A, mu_B, sigma = 40, 45, 4

    # Step 1: Likelihoods
    likelihood_A = gaussian_pdf(time, mu_A, sigma)
    likelihood_B = gaussian_pdf(time, mu_B, sigma)

    # Step 2: Total evidence (denominator)
    evidence = (likelihood_A * prior_A) + (likelihood_B * prior_B)

    # Step 3: Bayes rule → P(B | X = time)
    return (likelihood_B * prior_B) / evidence

def simulate_posterior_probability(time, n=100000):
    """
    Estimate P(B | X=time) using simulation.
    """
    sigma = 4

    # Simulate n swimmers, assign group by prior probability
    groups = np.random.choice(['A', 'B'], size=n, p=[0.3, 0.7])

    # Simulate finishing times based on group
    times = np.where(
        groups == 'A',
        np.random.normal(40, sigma, n),
        np.random.normal(45, sigma, n)
    )

    # Keep only swimmers near the observed time (tolerance window)
    tolerance = 0.5
    mask = np.abs(times - time) < tolerance

    # Among those near 'time', what fraction are from group B?
    if mask.sum() == 0:
        return 0.0  # edge case: no samples in window

    return np.mean(groups[mask] == 'B')
