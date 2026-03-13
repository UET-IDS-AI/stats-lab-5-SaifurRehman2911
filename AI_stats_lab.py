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
    return float(np.exp(-lam * a) - np.exp(-lam * b))


def simulate_exponential_probability(a, b, n=100000):
    """
    Simulate exponential samples and estimate P(a < X < b).
    """
    np.random.seed(42)
    samples = np.random.exponential(scale=1.0, size=n)
    return float(np.mean((samples > a) & (samples < b)))
    
# -------------------------------------------------
# Question 2 – Bayesian Classification
# -------------------------------------------------

def gaussian_pdf(x, mu, sigma):
    """
    Return Gaussian PDF.
    """
    coefficient = 1.0 / (sigma * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return float(coefficient * np.exp(exponent))


def posterior_probability(time):
    """
    Compute P(B | X = time) using Bayes rule.
    Priors:  P(A)=0.3, P(B)=0.7
    A ~ N(40, 4), B ~ N(45, 4)
    """
    prior_A, prior_B = 0.3, 0.7
    mu_A, mu_B, sigma = 40, 45, 4

    likelihood_A = gaussian_pdf(time, mu_A, sigma)
    likelihood_B = gaussian_pdf(time, mu_B, sigma)

    evidence = (likelihood_A * prior_A) + (likelihood_B * prior_B)

    return float((likelihood_B * prior_B) / evidence)


def simulate_posterior_probability(time, n=100000):
    """
    Estimate P(B | X=time) using simulation.
    """
    np.random.seed(42)
    sigma = 4

    # Assign groups by prior probability
    groups = np.random.choice([0, 1], size=n, p=[0.3, 0.7])  # 0=A, 1=B

    # Simulate finish times per group
    times = np.where(
        groups == 0,
        np.random.normal(40, sigma, n),
        np.random.normal(45, sigma, n)
    )

    # Tolerance window around observed time
    tolerance = 0.5
    mask = np.abs(times - time) < tolerance

    if mask.sum() == 0:
        return 0.0

    return float(np.mean(groups[mask] == 1))
