import numpy as np
from scipy.integrate import simpson
from scipy.stats import norm


# The Gaussian PDF callback function
def normal_pdf(x, mean, std):
    return norm.pdf(x, mean, std)


# The probability function that integrates the PDF
def probability(PDF, args, c, GT=False):
    mean, std = args
    # Create an array of x values from mean - 5*std to c for integration
    x = np.linspace(mean - 5 * std, c, 1000)
    # Calculate the corresponding y values using the PDF
    y = PDF(x, mean, std)
    # Use Simpson's rule for integration with keyword arguments
    area = simpson(y=y, x=x)
    return area if not GT else 1 - area


# The main function that uses the probability function to find specific probabilities
def main():
    # Calculate P(x<=105) with mu=100 and sigma=12.5
    prob_less_than_105 = probability(normal_pdf, (100, 12.5), 105, GT=False)
    # Calculate P(x>mu+2sigma) with mu=100 and sigma=3
    prob_greater_than_mu_plus_2sigma = probability(normal_pdf, (100, 3), 100 + 2 * 3, GT=True)

    # Print the results
    print(f'P(x<=105|N(100,12.5))={prob_less_than_105:.2f}')
    print(f'P(x>mu+2sigma|N(100, 3))={prob_greater_than_mu_plus_2sigma:.2f}')


# Execute the main function
if __name__ == "__main__":
    main()
