import math
from scipy.stats import norm

def calculate_greeks(option_type, S, K, T, r, iv):
    """
    Calculate Black-Scholes Greeks for European options.
    S: Spot price
    K: Strike price
    T: Time to expiry (in years)
    r: Risk-free rate (decimal)
    iv: Implied volatility (decimal, e.g., 0.18)
    option_type: 'CALL' or 'PUT'
    Returns: dict with Delta, Gamma, Theta, Vega, Rho
    """
    if T <= 0 or iv <= 0 or S <= 0 or K <= 0:
        return {g: 0.0 for g in ['Delta','Gamma','Theta','Vega','Rho']}
    d1 = (math.log(S/K) + (r + 0.5*iv**2)*T) / (iv*math.sqrt(T))
    d2 = d1 - iv*math.sqrt(T)
    if option_type == 'CALL':
        delta = norm.cdf(d1)
        theta = (-S*norm.pdf(d1)*iv/(2*math.sqrt(T)) - r*K*math.exp(-r*T)*norm.cdf(d2))/365
        rho = K*T*math.exp(-r*T)*norm.cdf(d2)/100
    else:
        delta = -norm.cdf(-d1)
        theta = (-S*norm.pdf(d1)*iv/(2*math.sqrt(T)) + r*K*math.exp(-r*T)*norm.cdf(-d2))/365
        rho = -K*T*math.exp(-r*T)*norm.cdf(-d2)/100
    gamma = norm.pdf(d1)/(S*iv*math.sqrt(T))
    vega = S*norm.pdf(d1)*math.sqrt(T)/100
    return {
        'Delta': delta,
        'Gamma': gamma,
        'Theta': theta,
        'Vega': vega,
        'Rho': rho
    }
