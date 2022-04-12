#!/usr/bin/env python
# coding: utf-8

# # Gaussian Basis Sets and Potentials (in one dimension)
# 
# ## &#x1f469;&#x1f3fd;&#x200d;&#x1f4bb; Assignment: Complete the Below Notebook
# 1. Complete the code blocks indicated by  `### START YOUR CODE HERE`  and  `### END YOUR CODE HERE` .
# 1. Upload your notebook *with the same name* and confirm its correctness. The marking scheme is provided at the end of this document.
# 1. There are *two* exercises and one bonus exercise in this notebook. The first has three parts; the second has two parts. The other portions of the notebook are explanation and utilities to help you with the exercises.
# 
# ## The Gaussian Well in One Dimension
# A (one-dimensional) Gaussian quantum well with dissociation energy $D$ is described by the Hamiltonian
# 
# $$
# \hat{H}_{\text{Gaussian}} = -\frac{\hbar^2}{2m} \frac{d^2}{dx^2} - D \cdot e^{-ax^2} 
# $$
# 
# Herein, we will assume that the bound particle is an electron, and use atomic units. The resulting Hamiltonian, if $a \ll 1$, can be well-approximated by the leading order terms in the Taylor expansion, 
# 
# $$
# \hat{H}_{\text{Gaussian}} = -\frac{1}{2} \frac{d^2}{dx^2} - D\left(1 - \tfrac{1}{2} a x^2 + \cdots \right) 
# $$
# 
# which is just a (shifted) harmonic oscillator
# 
# $$
# \hat{H}_{\text{shifted h.o.}} = -\frac{1}{2} \frac{d^2}{dx^2} - D +  \tfrac{1}{2} Da x^2 
# $$
# 
# In the first three parts of this problem, we will evaluate a zeroth-order approximation to the Gaussian quantum well based on this wavefunction, and compute more accurate corrections using perturbation theory and the variational principle.

# ## Useful Background Information

# ### Harmonic Oscillator Ground State
# The harmonic oscillator Hamiltonian has the form:
# 
# $$
# -\frac{\hbar^2}{2m} \frac{d^2}{dx^2} + \frac{1}{2}kx^2
# $$
# 
# The eigenvalues of the harmonic-oscillator Hamiltonian are:
# 
# $$
# E_n = \hbar \omega (n+\tfrac{1}{2}) \qquad  \qquad n=0,1,2,\ldots
# $$
# 
# where 
# 
# $$
# \omega = \sqrt{\tfrac{k}{m}}
# $$
# 
# is the angular frequency. The ground-state wavefunction is a [Gaussian function](https://en.wikipedia.org/wiki/Gaussian_function)
# 
# $$
# \psi_0(x) = \left(\frac{m \omega}{\hbar \pi} \right)^{\tfrac{1}{4}} e^{-\frac{m \omega x^2}{2 \hbar}}
# $$
# 
# Gaussian functions are ubiquitous in quantum chemistry because the integrals we need to evaluate are all relatively easy. A few special integrals that are especially useful are that a Gaussian multiplied by an odd power of $x$ always integrates to zero:
# 
# $$
# \int_{-\infty}^{\infty} x^{2k+1} e^{-ax^2} \, dx = 0 \qquad \qquad k=0,1,2,\ldots
# $$
# 
# while a Gaussian multiplied by an even power of $x$ has a quite simple formula
# 
# $$
# \int_{-\infty}^{\infty} x^{2k} e^{-ax^2} \, dx = \sqrt{\frac{\pi}{a}}\frac{(2k-1)(2k-3)\cdots(1)}{(2a)^k} \qquad \qquad k=1,2,3,\ldots
# $$
# 
# The most important cases of this are $k=0$, $k=2$, and $k=4$,
# 
# $$
# \begin{align}
# \int_{-\infty}^{\infty} e^{-ax^2} \, dx &= \sqrt{\frac{\pi}{a}} \\
# \int_{-\infty}^{\infty} x^2 e^{-ax^2} \, dx &= \frac{1}{2a}\sqrt{\frac{\pi}{a}} \\
# \int_{-\infty}^{\infty} x^4 e^{-ax^2} \, dx &= \frac{3}{4a^2}\sqrt{\frac{\pi}{a}} 
# \end{align}
# $$
# 
# The only other key integral one needs is related to the kinetic energy,
# 
# $$
# \begin{align}
# \int_{-\infty}^{\infty} e^{-ax^2} \frac{d^2}{dx^2} e^{-ax^2} \, dx &= \int_{-\infty}^{\infty} e^{-ax^2} \frac{d}{dx} (-2ax) e^{-ax^2} \, dx \\
# &= \int_{-\infty}^{\infty} e^{-ax^2} (-2a + 4a^2x^2) e^{-ax^2} \, dx \\
# &= \int_{-\infty}^{\infty} (-2a + 4a^2x^2)e^{-2ax^2} \, dx \\
# &= -\frac{2a\sqrt{\pi}}{\sqrt{2a}} + \frac{12a^2\sqrt{\pi}}{4a\sqrt{2a}} \\
# &=\sqrt{\frac{3a\pi}{2}}-\sqrt{2a\pi} \\
# &=-\sqrt{\frac{a \pi}{2}}
# \end{align}
# $$
# 

# ### Shifted Harmonic Oscillator Approximation
# An electron bound in a Gaussian potential well can be approximated as a shifted harmonic oscillator, plus a (hopefully small) correction. That is, we define;
# 
# $$
# \hat{H}(\lambda) = \hat{H}_{\text{shifted h.o.}} + \lambda (\hat{H}_{\text{Gaussian}}-\hat{H}_{\text{shifted h.o.}})
# $$
# 
# Obviously 
# 
# $$
# \hat{H}(0) = \hat{H}_{\text{shifted h.o.}} = -\frac{1}{2} \frac{d^2}{dx^2} - D +  \tfrac{1}{2} Da x^2
# $$
# 
# and
# 
# $$
# \frac{\partial \hat{H}}{\partial\lambda} = \hat{H}(1) - \hat{H}(0) = D \left(1 - \tfrac{1}{2} a x^2 -  e^{-ax^2} \right) 
# $$
# 
# We can evaluate the energy at zeroth-order (exercise 1a) and first-order (exercise 1b) using perturbation theory. We can also hypothesize that a better form for the wavefunction is the more general expression,
# 
# $$
# \psi_{\alpha}(x) = \left(\frac{2\alpha}{\pi} \right)^{\tfrac{1}{4}} e^{-\alpha x^2}
# $$
# 
# and use the variational principle to determine $\alpha$ (exercise 1c).
# 
# ## &#x1f469;&#x1f3fd;&#x200d;&#x1f4bb; **Exercise 1.** Approximate Treatment for a Single Well
# ### &#x1f469;&#x1f3fd;&#x200d;&#x1f4bb; **Exercise 1a.** Write a function to evaluate the ground-state energy when $\lambda=0$
# ### &#x1f469;&#x1f3fd;&#x200d;&#x1f4bb; **Exercise 1b.** Write a function to estimate the energy at $\lambda=1$, including the first-order correction from perturbation theory perturbation theory.
# ### &#x1f469;&#x1f3fd;&#x200d;&#x1f4bb; **Exercise 1c.** Use the variational principle to evaluate the best value for $\alpha$ in $\psi_{\alpha}(x)$ and define a function that returns the associated energy. 

# In[1]:


# Execute this code block to import required objects and compute a couple helpful utility functions.

# Note: The numpy library from autograd is imported, which behaves the same as
#       importing numpy directly. However, to make automatic differentiation work,
#       you should NOT import numpy directly by `import numpy as np`.

import autograd.numpy as np
from autograd import elementwise_grad as egrad, grad
from autograd import grad

import scipy
from scipy.integrate import quad
from scipy import constants

import matplotlib.pyplot as plt

# It is helpful to have code to evaluate the reference wavefunction. 
def psi(x, x0, alpha):
    """Compute ground state wavefunction for a 1D harmonic oscillator centered at x0.

    psi(x) = ((2*alpha)/pi)**(1/4) * exp(-alpha * (x - x0)**2)   
    
    Parameters
    ----------
    x: float or np.ndarray
        Position of the particle.
    x0: float
        Position of the bottom/center of the harmonic well.
    alpha: float 
        exponential parameter in the normalized GAussian wavefunction. 
        For the harm. osc., equal to 0.5 sqrt(k*m) where k is the spring constant and m is the mass. 
    """
    # check argument a
    if alpha <= 0.0:
        raise ValueError("Gaussian exponent, alpha, should be positive.")
    # check argument x0
    if not (isinstance(x0, float)):
        raise ValueError("The position of the well, x0, should be a float.")
    # check argument x
    if not (isinstance(x, float) or hasattr(x, "__iter__") or isinstance(x, np.numpy_boxes.ArrayBox)):
        raise ValueError("Position should be a float or an array.")
        
    # compute wave-function
    return ((2*alpha)/np.pi)**(1/4) * np.exp(-alpha * (x - x0)**2)

# Check normalization of the reference wavefunction.
def normalization(x0, alpha):
    """Check the normalization of the ground state wavefunction for a 1D harmonic oscillator centered at x0.

    integral over all space of psi(x)**2 = 1

    Parameters
    ----------
    x0: float
        Position of the bottom/center of the harmonic well.
    alpha: float 
        exponential parameter; equal to 0.5 sqrt(k*m) where k is the spring constant and m is the mass.
    """
    # check argument a
    if alpha <= 0.0:
        raise ValueError("Gaussian exponent, alpha, should be positive.")
    # check argument x0
    if not (isinstance(x0, float)):
        raise ValueError("The position of the well, x0, should be a float.")
    
    # compute normalization
    integrand = lambda x: psi(x, x0, alpha)**2
    ### END YOUR CODE HERE

    integral, error = quad(integrand,-np.inf,np.inf)

    return integral

print("Normalization of the reference wavefunction:", normalization(10.0, 2.5))


# In[24]:


np.random.seed(42)

answer = []
x = np.random.uniform(size=10)
x0 = np.random.randint(100, size=10)*1e-5
a = np.random.uniform(size=10)

for x_, x0_, a_ in zip(x, x0, a):
    answer.append(psi(x_, x0_, a_))
print(answer)

answer = []
x0 = np.random.randint(100, size=10)*1e-5
a = np.random.uniform(size=10)

for x0_, a_ in zip(x0, a):
    answer.append(normalization(x0_, a_))
    
answer


# > **Note:** You can do a little bit of mathematics that needs to be done before you can write functions for Exercises 1a-1c. Alternatively, you can use numerical integration and automatic differentiation. Choose your own adventure.

# In[2]:


# Exercise 1a. 
# Write a function to compute the zeroth-order energy
def compute_energy_0(D, a):
    """Compute energy of the shifted harmonic oscillator approximation to a Gaussian potential well.

    The form of the Gaussian potential well is 
        V(x) = -D * exp(-a * x^2)
    
    Parameters
    ----------
    D: float 
        Depth of the potential well.
    a: float 
        Controls width of the well.
    """
    # check argument D
    if D <= 0.0:
        raise ValueError("Depth of the well should be postive.")
    # check argument a
    if a <= 0.0:
        raise ValueError("Width parameter for the well should be postive.")

    # compute and return the value of the zeroth-order energy.
    ### START YOUR CODE HERE 
    
    return 1/2 * np.sqrt(D*a) - D

    ### END YOUR CODE HERE


# In[27]:


answer = []
np.random.seed(42)
D = np.random.randint(100, size=10) * 1e-5
a = np.random.uniform(size=10)

for D_, a_ in zip(D, a):
    answer.append(compute_energy_0(D_, a_))

answer


# In[3]:


# Exercise 1b.
# Complete the function to compute the first-order correction to the energy.
# Then compute the energy correct to first order (given code).
def compute_energy_1(D, a):
    """Compute first-order correction to the energy of a shifted h.o. for the Gaussian well.
    
    Parameters
    ----------
    D: float 
        Depth of the potential well.
    a: float 
        Controls width of the well.
    """
    # check argument D
    if D <= 0.0:
        raise ValueError("Depth of the well should be postive.")
    # check argument a
    if a <= 0.0:
        raise ValueError("Width parameter for the well should be postive.")

    # compute and return the value of the integral one needs for the first-order perturbative correction.
    ### START YOUR CODE HERE
    # The Gaussian is centered at the origin
    x0 = 0.0
    # The width of the Gaussian wavefunction is:
    alpha = np.sqrt(D*a)/2 

    # compute integrand for the perturbation
    integrand = lambda x: psi(x, x0, alpha)**2 * D*(1 - a*x**2/2 - np.exp(-a*x**2))

    integral, error = quad(integrand,-np.inf,np.inf)

    return integral    

def compute_energy_1_analytic(D, a):
    """ Analytic computation of the first-order correction to the energy.
    
    Parameters
    ----------
    D: float 
        Depth of the potential well.
    a: float 
        Controls width of the well.
    """
    # check argument D
    if D <= 0.0:
        raise ValueError("Depth of the well should be postive.")
    # check argument a
    if a <= 0.0:
        raise ValueError("Width parameter for the well should be postive.")

    # compute first-order correction using analytic expression
    # it's useful to define the parameter alpha (exponent of the Gaussian)
    alpha = np.sqrt(D*a)/2

    value = D * (1 - a/(8*alpha) - np.sqrt((2*alpha)/(2*alpha + a)))

    return value

#Compare the analytic and numerical results as a test.
print ("Energy of the first-order correction:", compute_energy_1(1.0, 2.0))
print ("Energy of the first-order correction analytic:", compute_energy_1_analytic(1.0, 2.0))
print ("Energy of the first-order correction:", compute_energy_1(1.0, 0.1))
print ("Energy of the first-order correction analytic:", compute_energy_1_analytic(1.0, 0.1))

    ### END YOUR CODE HERE
    
# The following function computes the energy of the Gaussian potential well, accurate to first order.
def compute_energy_01(D,a):
    """Compute the energy of the Gaussian well, correct to first order.
    """
    return compute_energy_0(D,a) + compute_energy_1(D,a)  


# In[31]:


np.random.seed(42)
D = np.random.randint(100, size=10) * 1e-5
a = np.random.uniform(size=10)

answer = []
for D_, a_ in zip(D, a):
    answer.append(compute_energy_1(D_, a_))
    
answer


# In[4]:


# Exercise 1c.
# Write a function to evaluate the expectation value of the variational energy expression
# Then determine the best value of alpha (given code).

def compute_expected_energy(alpha, D, a):
    """Compute the expectation value of the energy of the Gaussian well, with a trial wf of width alpha.
    
    Parameters
    ----------
    D: float 
        Depth of the potential well.
    a: float 
        Controls width of the well.
    alpha: float
        Controls width of the Gaussian.
    """
    # check argument D
    if D <= 0.0:
        raise ValueError("Depth of the well should be postive.")
    # check argument a
    if a <= 0.0:
        raise ValueError("Width parameter for the well should be postive.")
    # check argument alpha
    if alpha <= 0.0:
        raise ValueError("Width parameter for the Gaussian should be postive.")
    
    # Assume the Gaussian and the potential well are centered at the origin
    x0 = 0.0

    # compute and return the expected energy as a function of the trial wavefunction parameter, alpha.
    ### START YOUR CODE HERE

    # compute integrand for the perturbation
    integrand = lambda x: psi(x, x0, alpha)**2 * (alpha - 2*alpha**2*x**2 - D*np.exp(-a*x**2))

    integral, error = quad(integrand,-np.inf,np.inf)

    return integral  

def wf_derivative(x, x0, alpha, order=1):
    """Compute the derivative of a s-type normalized Gaussian in one dimension.
    """
    if not (isinstance(order, int) and order > 0):
        raise ValueError("The order of differentiation should be a positive integer!")

    def wavefunction(x):
        v = psi(x, x0, alpha)
        return v

    # compute derivative
    deriv = egrad(wavefunction)
    for _ in range(order - 1):
        deriv = egrad(deriv)

    deriv = deriv(x)
  
    return deriv

def compute_expected_energy_autodiff(alpha,D, a):
    """Compute the expectation value of the energy of the Gaussian well, with a trial wf of width alpha.
    
    This version uses automatic differentiation.

    Parameters
    ----------
    D: float 
        Depth of the potential well.
    a: float 
        Controls width of the well.
    alpha: float
        Controls width of the Gaussian.
    """
    # check argument D
    if D <= 0.0:
        raise ValueError("Depth of the well should be postive.")
    # check argument a
    if a <= 0.0:
        raise ValueError("Width parameter for the well should be postive.")
    # check argument alpha
    if alpha <= 0.0:
        raise ValueError("Width parameter for the Gaussian should be postive.")
    
    # Assume the Gaussian and the potential well are centered at the origin
    x0 = 0.0
    
    # compute and return the expected energy as a function of the trial wavefunction parameter, alpha.
    # compute integrand for the perturbation
    integrand = lambda x: psi(x, x0, alpha)*(-0.5*wf_derivative(x,x0,alpha,order=2) 
                            - D*np.exp(-a*x**2)*psi(x, x0, alpha))
  
    integral, error = quad(integrand,-np.inf,np.inf)

    return integral

def compute_expected_energy_analytic(alpha,D, a):
    """Compute the expectation value of the energy of the Gaussian well, with a trial wf of width alpha.
    
    This version uses analytic expressions.

    Parameters
    ----------
    D: float 
        Depth of the potential well.
    a: float 
        Controls width of the well.
    alpha: float
        Controls width of the Gaussian.
    """
    # check argument D
    if D <= 0.0:
        raise ValueError("Depth of the well should be postive.")
    # check argument a
    if a <= 0.0:
        raise ValueError("Width parameter for the well should be postive.")
    # check argument alpha
    if alpha <= 0.0:
        raise ValueError("Width parameter for the Gaussian should be postive.")
    
    # Assume the Gaussian and the potential well are centered at the origin
    x0 = 0.0
    
    # compute and return the expected energy as a function of the trial wavefunction parameter, alpha.

    return alpha/2 - D * np.sqrt((2*alpha)/(2*alpha + a))

#print(wf_derivative(1.0, 1.0, 1.0, order=1))   #This is giving an error.
print ("Energy of the variational energy expression numer.:   ", compute_expected_energy(1.0, 2.0, 0.1))
print ("Energy of the variational energy expression autodiff.:", compute_expected_energy_autodiff(1.0, 2.0, 0.1))
print ("Energy of the variational energy expression explicit: ", compute_expected_energy_analytic(1.0, 2.0, 0.1))
print ("Energy of the variational energy expression numer.:   ", compute_expected_energy(4.0, 1.0, 1.0))
print ("Energy of the variational energy expression autodiff.:", compute_expected_energy_autodiff(4.0, 1.0, 1.0))
print ("Energy of the variational energy expression explicit: ", compute_expected_energy_analytic(4.0, 1.0, 1.0))

    ### END YOUR CODE HERE

def energy_variational(D,a,alpha0=None):
    """Compute the energy of the Gaussian well, correct to first order.

    Parameters
    ----------
    D: float 
        Depth of the potential well.
    a: float 
        Controls width of the well.
    alpha0: float
        Initial guess for the width of the Gaussian; if not given, zeroth-order
        guess from shifted harmonic oscillator
    """
    
    if alpha0 is None:
        alpha0 = np.sqrt(D*a)/2

    # minimize the energy
    result = scipy.optimize.minimize(compute_expected_energy, alpha0, (D,a))

    # return the optimized energy
    return result.fun



# In[39]:


np.random.seed(42)
alpha = np.random.randint(0, 10)
D = np.random.randint(100, size=10)*1e-2
a = np.random.uniform(size=10)

answer = []
for D_, a_ in zip(D, a):
    answer.append(compute_expected_energy(alpha, D_, a_))
    
answer


# In[5]:


# This code block prints a summary of your results.
print("Zeroth order energy from the shifted harmonic oscillator:", compute_energy_0(4.0, 1.0))
print("Energy from perturbation theory, through first order:", compute_energy_01(4.0, 1.0))
print("Energy of the variational energy expression:", compute_expected_energy(1.0, 4.0, 1.0))
print("Variationally optimized energy:", energy_variational(4.0, 1.0))
print("Zeroth order energy from the shifted harmonic oscillator:", compute_energy_0(1.0, 0.5))
print("Energy from perturbation theory, through first order:", compute_energy_01(1.0, 0.5))
print("Energy of the variational energy expression:", compute_expected_energy(0.3536, 1.0, 0.5))
print("Variationally optimized energy:", energy_variational(1.0, 0.5))


# ### Answers (Notes for solutions.)
# ### Exercise 1a. 
# The shifted-harmonic-oscillator Hamiltonian corresponds to the harmonic oscillator with $m=1$ and $k=Da$. Then $D$ is subtracted from the energy. So 
# 
# $$
# E_{\lambda=0}(D,a) = \tfrac{1}{2}\sqrt{Da} - D
# $$
# 
# ### Exercise 1b. 
# The zeroth-order wavefunction has $\alpha = \frac{\sqrt{k}}{2} = \frac{\sqrt{Da}}{2}$. 
# The integral we need to evaluate is: 
# 
# $$
# \begin{align}
# \int_{-\infty}^{\infty} &\psi_{\alpha}(x) D \left(1 - \tfrac{1}{2} a x^2 -  e^{-ax^2} \right)  \psi_{\alpha}(x) \, dx = \left(\frac{2\alpha}{\pi} \right)^{\tfrac{1}{2}}D \int_{-\infty}^{\infty} e^{-\alpha x^2} \left(1 - \tfrac{1}{2} a x^2 -  e^{-ax^2} \right) e^{-\alpha x^2} \, dx \\
# &= D - \frac{Da}{2} \left(\frac{2\alpha}{\pi} \right)^{\tfrac{1}{2}}  \int_{-\infty}^{\infty} x^2 e^{-2\alpha x^2} \, dx - D\left(\frac{2\alpha}{\pi} \right)^{\tfrac{1}{2}}\int_{-\infty}^{\infty} e^{-(2\alpha + a) x^2} \, dx \\
# &= D - \frac{Da}{2} \left(\frac{2\alpha}{\pi} \right)^{\tfrac{1}{2}}\frac{1}{4 \alpha}\sqrt{\frac{\pi}{2 \alpha}}   - D\left(\frac{2\alpha}{\pi} \right)^{\tfrac{1}{2}} \sqrt{\frac{\pi}{2 \alpha+a}} \\
# &= D \left(1 - \frac{a}{8 \alpha} - \sqrt{\frac{2 \alpha}{2 \alpha+a}} \right)
# \end{align}
# $$
# 
# Using $\alpha = \frac{\sqrt{Da}}{2}$,
# $$
# \begin{align}
# \int_{-\infty}^{\infty} &\psi_{\alpha}(x) D \left(1 - \tfrac{1}{2} a x^2 -  e^{-ax^2} \right)  \psi_{\alpha}(x) \, dx = D \left(1 - \frac{2a}{4 \sqrt{Da}} - \sqrt{\frac{\sqrt{Da}}{\sqrt{Da}+a}} \right) \\
# &= D - \tfrac{1}{2}\sqrt{Da} - D  \sqrt{\frac{\sqrt{Da}}{\sqrt{Da}+a}} 
# \end{align}
# $$
# 
# ### Exercise 1c. 
# The integral we need to evaluate is:
# $$
# \begin{align}
# \int_{-\infty}^{\infty} \psi_{\alpha}(x) \hat{H}_{\text{Gaussian}} \psi_{\alpha}(x) \, dx &= \left(\frac{2\alpha}{\pi} \right)^{\tfrac{1}{2}}\int_{-\infty}^{\infty} e^{-\alpha x^2} \left(-\frac{1}{2} \frac{d^2}{dx^2} - D \cdot e^{-ax^2}  \right) e^{-\alpha x^2} \, dx \\
# &=\left(\frac{2\alpha}{\pi} \right)^{\tfrac{1}{2}}\int_{-\infty}^{\infty} e^{-\alpha x^2} \left(\alpha - 2\alpha^2x^2 - D \cdot e^{-ax^2}  \right) e^{-\alpha x^2} \, dx \\
# &=\left(\frac{2\alpha}{\pi} \right)^{\tfrac{1}{2}}\int_{-\infty}^{\infty} \left(\alpha - 2\alpha^2x^2  \right) e^{-2\alpha x^2} \, dx - D \int_{-\infty}^{\infty} e^{-(2\alpha+a) x^2} \, dx\\
# &=\left(\frac{2\alpha}{\pi} \right)^{\tfrac{1}{2}}\left(\alpha \sqrt{\frac{\pi}{2\alpha}} - 2 \alpha^2 \frac{1}{4\alpha}\sqrt{\frac{\pi}{2\alpha}} -D   \sqrt{\frac{\pi}{2\alpha+a}}\right) \\
# &=\tfrac{1}{2}\alpha - D \sqrt{\frac{2 \alpha}{2 \alpha + a}} 
# \end{align}
# $$
# We can then differentiat this expression,
# $$
# \begin{align}
# 0 &=\frac{dE(\alpha)}{d\alpha}E(\alpha) = \frac{d}{d\alpha}\left(\tfrac{1}{2}\alpha - D \sqrt{\frac{2 \alpha}{2 \alpha + a}} \right) \\
# &=\tfrac{1}{2} -\tfrac{1}{2}D\sqrt{\frac{2 \alpha + a}{2 \alpha}}\left(\frac{2(2\alpha + a)-4\alpha}{(2\alpha+a)^2}\right) \\
# &=\tfrac{1}{2} -\tfrac{1}{2}D\sqrt{\frac{2a}{2 \alpha(2 \alpha + a)^3}} \\
# D^{-1}&=\sqrt{\frac{2a}{2 \alpha(2 \alpha + a)^3}} \\
# \sqrt{2aD^2} = 2 \alpha(2 \alpha + a)^3 \\
# 0 &= 2 \alpha(2 \alpha + a)^3 - \sqrt{2aD^2}
# \end{align}
# $$
# There is an analytic solution to this (quartic) equation but it is painful to evaluate.

# ## *N* Electrons in a Multi-Well Gaussian Potential
# An electron in a Gaussian well can be thought of as a model for an atom with one valence electron. The corresponding model for $N$ noninteracting electrons in a linear molecule (e.g., a $\pi$-conjugated system) with $P$ atoms is: 
# 
# $$
# \hat{H}_{\text{multi}} = \sum_{i=0}^{N-1} \left(-\frac{1}{2} \frac{d^2}{dx_i^2} - \sum_{A=0}^{P-1} D_A \cdot e^{-a_A(x_i-x_A)^2} \right)
# $$
# 
# For simplicity, we'll use as a trial wavefunction the linear combination of "atomic" orbitals,
# 
# $$
# \psi(x) = \sum_{A=0}^{P-1} c_A \phi_A(x)
# $$
# 
# and assume that the shifted-harmonic-oscillator approximation is not too terrible, so 
# 
# $$
# \phi_A(x) = \left(\frac{2\alpha_A}{\pi} \right)^{\tfrac{1}{4}} e^{-\alpha_A (x-x_A)^2}
# $$
# 
# where the value of $\alpha$ is the taken from the shifted harmonic oscillator (cf. your answer in exercise 1a).
# 
# ### Gaussian Product Rule
# One reason that Gaussian basis functions are so ubiquitous in chemistry is that the integrals one needs are often easy. One reason they are easy is that the product of two Gaussians is just another Gaussian. 
# $$
# e^{-\alpha_A (x-x_A)^2}e^{-\alpha_B (x-x_B)^2} = e^{-\frac{p_{AB}(x_A-x_B)^2}{s_{AB}}}e^{-s_{AB}(x-x_{AB})^2}
# $$
# where
# $$
# \begin{align}
# p_{AB} &= \alpha_A \alpha_B \\
# s_{AB} &= \alpha_A + \alpha_B \\
# x_{AB} &= \frac{\alpha_A x_A + \alpha_B x_B}{s_{AB}}
# \end{align}
# $$
# 
# There is a similar, slightly more obscure, rule for the product of three Gaussians:
# $$
# e^{-\alpha_A (x-x_A)^2}e^{-\alpha_B (x-x_B)^2}e^{-\alpha_C (x-x_C)^2} = e^{-\frac{p_{AB}(x_A-x_B)^2}{s_{ABC}}}e^{-\frac{p_{BC}(x_B-x_C)^2}{s_{ABC}}}e^{-\frac{p_{AC}(x_A-x_C)^2}{s_{ABC}}}e^{-s_{ABC}(x-x_{ABC})^2}
# $$
# where
# $$
# \begin{align}
# s_{ABC} &= \alpha_A + \alpha_B + \alpha_C\\
# x_{ABC} &= \frac{\alpha_A x_A + \alpha_B x_B + \alpha_C x_C}{s_{ABC}}
# \end{align}
# $$

# ### Useful integrals
# 
# #### Overlap Integrals
# $$
# \begin{align}
# S_{AB} &= \int_{-\infty}^{+\infty} \phi_A(x) \phi_B(x) \, dx \\
# &=\left(\frac{4\alpha_A \alpha_B}{\pi^2} \right)^{\tfrac{1}{4}}\int_{-\infty}^{+\infty} e^{-\alpha_A (x-x_A)^2}e^{-\alpha_B (x-x_B)^2} \, dx \\
# &=\left(\frac{4 p_{AB}}{\pi^2} \right)^{\tfrac{1}{4}} e^{-\tfrac{p_{AB}}{s_{AB}}(x_A-x_B)^2}\int_{-\infty}^{+\infty}e^{-s_{AB}(x-x_{AB})^2} \, dx \\
# &=\left(\frac{4 p_{AB}}{\pi^2} \right)^{\tfrac{1}{4}} e^{-\tfrac{p_{AB}}{s_{AB}}(x_A-x_B)^2} \sqrt{\frac{\pi}{s_{AB}}} \\
# &=\left(\frac{4p_{AB}}{s_{AB}^2} \right)^{\tfrac{1}{4}} e^{-\frac{p_{AB}(x_A-x_B)^2}{s_{AB}}}
# \end{align}
# $$

# #### Potential Integrals
# $$
# \begin{align}
# V_{AB} &=  \int_{-\infty}^{+\infty} \phi_A(x) \hat{V} \phi_B(x) \, dx \\
# &=-\sum_{C=0}^{P-1} D_C \int_{-\infty}^{+\infty} \phi_A(x) e^{-a_C(x-x_C)^2} \phi_B(x) \, dx \\
# &=-\sum_{C=0}^{P-1} D_C \left(\frac{4\alpha_A \alpha_B}{\pi^2} \right)^{\tfrac{1}{4}} \int_{-\infty}^{+\infty} e^{-\alpha_A (x-x_A)^2}e^{-\alpha_B (x-x_B)^2} e^{-a_C(x-x_C)^2}  \, dx \\
# &=-\sum_{C=0}^{P-1} D_C \left(\frac{4p'_{AB}}{\pi^2} \right)^{\tfrac{1}{4}}e^{-\frac{p'_{AB}(x_A-x_B)^2}{s'_{ABC}}}e^{-\frac{p'_{BC}(x_B-x_C)^2}{s'_{ABC}}}e^{-\frac{p'_{AC}(x_A-x_C)^2}{s'_{ABC}}}\int_{-\infty}^{+\infty} e^{-s'_{ABC}(x-x_{ABC})^2} \, dx \\
# &=-\sum_{C=0}^{P-1} D_C \left(\frac{4p'_{AB}}{(s'_{ABC})^2} \right)^{\tfrac{1}{4}}e^{-\frac{p'_{AB}(x_A-x_B)^2}{s'_{ABC}}}e^{-\frac{p'_{BC}(x_B-x_C)^2}{s'_{ABC}}}e^{-\frac{p'_{AC}(x_A-x_C)^2}{s'_{ABC}}} 
# \end{align}
# $$
# where $p'_{AC}$ and $s'_{ABC}$ are defined as above, but with $\alpha_C$ replaced by $a_C$.

# #### Kinetic Energy Integrals
# For the kinetic energy integral, it is useful to remember that, by using integration by parts, one can compute the kinetic energy in either of the following two forms:
# $$
# \begin{align}
# \int_{-\infty}^{+\infty} \phi_A(x) \hat{T} \phi_B(x) \, dx &= \int_{-\infty}^{+\infty} \phi_A(x) \left( \tfrac{1}{2} \tfrac{d^2}{dx^2}\right) \phi_B(x) \, dx \\
# &=\tfrac{1}{2}\int_{-\infty}^{+\infty} \left( \tfrac{d}{dx}\phi_A(x)\right)  \left( \tfrac{d}{dx}\phi_B(x)\right)  \, dx 
# \end{align}
# $$
# The latter form is more convenient here.
# $$
# \begin{align}
# T_{AB} &= \frac{1}{2}\int_{-\infty}^{+\infty} \left( \tfrac{d}{dx}\phi_A(x)\right)  \left( \tfrac{d}{dx}\phi_B(x)\right)  \, dx\\
# &=\frac{1}{2}\left(\frac{4\alpha_A \alpha_B}{\pi^2} \right)^{\tfrac{1}{4}}
# \int_{-\infty}^{+\infty}\left( \tfrac{d}{dx}e^{-\alpha_A (x-x_A)^2}\right)  \left( \tfrac{d}{dx}e^{-\alpha_B (x-x_B)^2}\right) \, dx \\
# &=\frac{1}{2}\left(\frac{4\alpha_A \alpha_B}{\pi^2} \right)^{\tfrac{1}{4}} \left(4\alpha_A \alpha_B \right) \int_{-\infty}^{+\infty}e^{-\alpha_A (x-x_A)^2}(x-x_A)(x-x_B) e^{-\alpha_B (x-x_B)^2} \, dx \\
# &=\left(\frac{4\alpha_A \alpha_B}{\pi^2} \right)^{\tfrac{1}{4}} \left(2\alpha_A \alpha_B \right) \int_{-\infty}^{+\infty}e^{-\alpha_A (x-x_A)^2}(x^2 - (x_A+x_B)x + x_A x_B) e^{-\alpha_B (x-x_B)^2} \, dx \\
# &=\left(\frac{4p_{AB}}{\pi^2} \right)^{\tfrac{1}{4}} \left(2p_{AB} \right)  e^{-\frac{p_{AB}(x_A-x_B)^2}{s_{AB}}}\int_{-\infty}^{+\infty}(x^2 - (x_A+x_B)x + x_A x_B) e^{-s_{AB}(x-x_{AB})^2} \, dx
# \end{align}
# $$
# 
# To evaluate the integral, we need to define a new variable, $u=x-x_{AB}$. Then 
# $$
# x = u + x_{AB}
# $$
# and the expression becomes: 
# $$
# (u+x_{AB})^2 - (x_A+x_B)(u+x_{AB}) + x_A x_B) = u^2 + (2x_{AB} -x_A - x_B)u + (x_{AB}^2 - (x_A+x_B)x_{AB} + x_Ax_B
# $$
# So the integral can be simplified to: 
# $$
# \begin{align}
# T_{AB} &=\left(\frac{4p_{AB}}{\pi^2} \right)^{\tfrac{1}{4}} \left(2p_{AB} \right)  e^{-\frac{p_{AB}(x_A-x_B)^2}{s_{AB}}} \int_{-\infty}^{+\infty}(u^2 + (2x_{AB} -x_A - x_B)u + (x_{AB}^2 - (x_A+x_B)x_{AB} + x_Ax_B) e^{-s_{AB}u^2} \, dx \\
# &=\left(\frac{4p_{AB}}{\pi^2} \right)^{\tfrac{1}{4}} \left(2p_{AB} \right)  e^{-\frac{p_{AB}(x_A-x_B)^2}{s_{AB}}} \left(\frac{1}{2s_{AB}}\sqrt{\frac{\pi}{s_{AB}}} + (x_{AB}^2 - (x_A+x_B)x_{AB} + x_Ax_B)\sqrt{\frac{\pi}{s_{AB}}} \right) \\
# &=\left(\frac{4p_{AB}}{s_{AB}^2} \right)^{\tfrac{1}{4}} \left(2p_{AB} \right)  e^{-\frac{p_{AB}(x_A-x_B)^2}{s_{AB}}} \left(\frac{1}{2s_{AB}} + (x_{AB}^2 - (x_A+x_B)x_{AB} + x_Ax_B) \right)
# \end{align}
# $$

# In[6]:


# The following code implements (and checks) the above integrals.

def compute_S(x0, alpha, n_basis):
    """Compute the overlap matrix <m|n> for 1-dimensional Gaussian wavefunctions.
    
    Parameters
    ----------
    x0: float or ndarray of size n_basis
        Position of the Gaussian center. All wells are
        assumed to be co-located if this is a float.
    alpha: float or ndarray of size n_basis
        exponential parameter in the normalized Gaussian wavefunction.
        If float; all Gaussians are assumed to have the same exponent.
    n_basis : scalar, int
        the number of Gaussian wavefunctions to be considered.

    Returns
    -------
    S : array-like (nbasis, nbasis)
         Overlap matrix elements S_mn = <m | n> for the Gaussian wavefunctions.
         
    """
    # check argument n_basis
    if n_basis <= 0:
        raise ValueError("The number of basis functions should be positive.")
    
    # check argument alpha
    if isinstance(alpha, float):
        alpha = np.full(n_basis, alpha)
    elif isinstance(alpha, np.ndarray):
        if alpha.size != n_basis:
            raise ValueError("The number of basis functions should be the same as the length of the array alpha.")
    else:
        raise TypeError("The width parameter should be a float or an array.")
    
    # check argument x0
    if isinstance(x0, float):
        x0 = np.full(n_basis, x0)
    elif isinstance(x0, np.ndarray):
        if x0.size != n_basis:
            raise ValueError("The number of basis functions should be the same as the length of the array x0.")
            
    else:
        raise TypeError("The centers, x0, parameter should be a float or an array.")

    # initialize S to a zero matrix
    S = np.zeros((n_basis,n_basis))
    
    for n in range(n_basis):
        S[n,n] = 1.0
        for m in range(n):
            s = alpha[m]+alpha[n]
            p = alpha[m]*alpha[n]
            S[m,n] = (4*p/s**2)**(1/4)*np.exp(-p/s*(x0[m]-x0[n])**2)
            S[n,m] = S[m,n]
    
    return S

def compute_S_numer(x0, alpha, n_basis):
    """Compute the overlap matrix <m|n> for 1-dimensional Gaussian wavefunctions by quadrature.
    
    Parameters
    ----------
    x0: float or ndarray of size n_basis
        Position of the Gaussian center. All wells are
        assumed to be co-located if this is a float.
    alpha: float or ndarray of size n_basis
        exponential parameter in the normalized Gaussian wavefunction.
        If float; all Gaussians are assumed to have the same exponent.
    n_basis : scalar, int
        the number of Gaussian wavefunctions to be considered.

    Returns
    -------
    S : array-like (nbasis, nbasis)
         Overlap matrix elements S_mn = <m | n> for the Gaussian wavefunctions.
         
    """
        # check argument n_basis
    if n_basis <= 0:
        raise ValueError("The number of basis functions should be positive.")
    
    # check argument alpha
    if isinstance(alpha, float):
        alpha = np.full(n_basis, alpha)
    elif isinstance(alpha, np.ndarray):
        if alpha.size != n_basis:
            raise ValueError("The number of basis functions should be the same as the length of the array alpha.")
    else:
        raise TypeError("The width parameter should be a float or an array.")
    
    # check argument x0
    if isinstance(x0, float):
        x0 = np.full(n_basis, x0)
    elif isinstance(x0, np.ndarray):
        if x0.size != n_basis:
            raise ValueError("The number of basis functions should be the same as the length of the array x0.")
            
    else:
        raise TypeError("The centers, x0, parameter should be a float or an array.")

    # initialize S to a zero matrix
    S = np.zeros((n_basis,n_basis))
    for n in range(n_basis):
        S[n,n] = 1.0
        for m in range(n):
            integrand = lambda x: psi(x, x0[m], alpha[m])*psi(x, x0[n], alpha[n])
            integral, error = quad(integrand,-np.inf,np.inf)
            S[m,n] = integral
            S[n,m] = S[m,n]
    
    return S

def compute_T(x0, alpha, n_basis):
    """Compute the kinetic energy <m|T|n> for 1-dimensional Gaussian wavefunctions.
    
    Parameters
    ----------
    x0: float or ndarray of size n_basis
        Position of the Gaussian center. All wells are
        assumed to be co-located if this is a float.
    alpha: float or ndarray of size n_basis
        exponential parameter in the normalized Gaussian wavefunction.
        If float; all Gaussians are assumed to have the same exponent.
    n_basis : scalar, int
        the number of Gaussian wavefunctions to be considered.

    Returns
    -------
    T : array-like (nbasis, nbasis)
         K.E. matrix elements T_mn = <m | T | n> for the Gaussian wavefunctions.
         
    """
    # check argument n_basis
    if n_basis <= 0:
        raise ValueError("The number of basis functions should be positive.")
    
    # check argument alpha
    if isinstance(alpha, float):
        alpha = np.full(n_basis, alpha)
    elif isinstance(alpha, np.ndarray):
        if alpha.size != n_basis:
            raise ValueError("The number of basis functions should be the same as the length of the array alpha.")
    else:
        raise TypeError("The width parameter should be a float or an array.")
    
    # check argument x0
    if isinstance(x0, float):
        x0 = np.full(n_basis, x0)
    elif isinstance(x0, np.ndarray):
        if x0.size != n_basis:
            raise ValueError("The number of basis functions should be the same as the length of the array x0.")   
    else:
        raise TypeError("The centers, x0, parameter should be a float or an array.")

    # initialize T to a zero matrix
    T = np.zeros((n_basis,n_basis))
    
    for n in range(n_basis):
        T[n,n] = 0.5*alpha[n]
        for m in range(n):
            s = alpha[m]+alpha[n]
            p = alpha[m]*alpha[n]
            xAB = (alpha[m]*x0[m]+alpha[n]*x0[n])/s
            T[m,n] = (4*p/s**2)**(1/4)*2*p*np.exp(-p/s*(x0[m]-x0[n])**2)*(1/(2*s) + xAB**2
                        -xAB*(x0[m]+x0[n]) + x0[m]*x0[n])
            T[n,m] = T[m,n]
    
    return T

def compute_T_numer(x0, alpha, n_basis):
    """Compute the kinetic energy matrix <m|T|n> for 1-dimensional Gaussian wavefunctions by quadrature.
    
    Parameters
    ----------
    x0: float or ndarray of size n_basis
        Position of the Gaussian center. All wells are
        assumed to be co-located if this is a float.
    alpha: float or ndarray of size n_basis
        exponential parameter in the normalized Gaussian wavefunction.
        If float; all Gaussians are assumed to have the same exponent.
    n_basis : scalar, int
        the number of Gaussian wavefunctions to be considered.

    Returns
    -------
    T : array-like (nbasis, nbasis)
         K.E. matrix elements T_mn = <m | T | n> for the Gaussian wavefunctions.
         
    """
        # check argument n_basis
    if n_basis <= 0:
        raise ValueError("The number of basis functions should be positive.")
    
    # check argument alpha
    if isinstance(alpha, float):
        alpha = np.full(n_basis, alpha)
    elif isinstance(alpha, np.ndarray):
        if alpha.size != n_basis:
            raise ValueError("The number of basis functions should be the same as the length of the array alpha.")
    else:
        raise TypeError("The width parameter should be a float or an array.")
    
    # check argument x0
    if isinstance(x0, float):
        x0 = np.full(n_basis, x0)
    elif isinstance(x0, np.ndarray):
        if x0.size != n_basis:
            raise ValueError("The number of basis functions should be the same as the length of the array x0.")
            
    else:
        raise TypeError("The centers, x0, parameter should be a float or an array.")
    
    # initialize T to a zero matrix
    T = np.zeros((n_basis,n_basis))
    for n in range(n_basis):
        integrand = lambda x: psi(x, x0[n], alpha[n])**2*(alpha[n]**2)*(x-x0[n])**2
        integral, error = quad(integrand,-np.inf,np.inf)
        T[n,n] = integral
        for m in range(n):
            integrand = lambda x: psi(x, x0[m], alpha[m])*psi(x, x0[n], alpha[n])*(2*
                                        (alpha[m]*alpha[n])*(x-x0[m])*(x-x0[n]))
            integral, error = quad(integrand,-np.inf,np.inf)
            T[m,n] = integral
            T[n,m] = T[m,n]   
    return T

def compute_V(x0, alpha, n_basis, DA, aA, xA, n_sites):
    """Compute the potential energy <m|V|n> for 1-dimensional Gaussian wavefunctions.

    V = sum(0,1,2,..n_sites) -DA exp(-aA*(x-xA)**2)
    
    Parameters
    ----------
    x0 : float or ndarray of size n_basis
        Position of the Gaussian center. All wells are
        assumed to be co-located if this is a float.
    alpha : float or ndarray of size n_basis
        exponential parameter in the normalized Gaussian wavefunction.
        If float; all Gaussians are assumed to have the same exponent.
    n_basis : scalar, int
        the number of Gaussian wavefunctions to be considered.
    DA : float or ndarray of size n_sites
        Depth of the potential well. All wells are co-located if this
        is a float.
    xA : float or ndarray of size n_sites
        Position of the potential well. All wells are co-located if this
        is a float.
    aA : float or ndarray of size n_sites
        exponential parameter controlling the width of the Gaussian well.
        If float; all wells have the same width. The half-width at half-maximum
        is sqrt((ln 2)/a) and the zero-point energy in the harmonic oscillator
        approximation is 1/2 sqrt(D * a).
    n_sites : scalar, int
        the number sites in the multi-potential well.   

    Returns
    -------
    V : array-like (nbasis, nbasis)
         Pot. E. matrix elements V_mn = <m | V | n> for the Gaussian wavefunctions.
         
    """
    # check argument n_basis
    if n_basis <= 0:
        raise ValueError("The number of basis functions should be positive.")
    
    # check argument n_sites
    if n_sites <= 0:
        raise ValueError("The number of sites should be positive.")
    
    # check argument alpha
    if isinstance(alpha, float):
        alpha = np.full(n_basis, alpha)
    elif isinstance(alpha, np.ndarray):
        if alpha.size != n_basis:
            raise ValueError("The number of basis functions should be the same as the length of the array alpha.")
    else:
        raise TypeError("The Gaussian width parameter should be a float or an array.")
    
    # check argument x0
    if isinstance(x0, float):
        x0 = np.full(n_basis, x0)
    elif isinstance(x0, np.ndarray):
        if x0.size != n_basis:
            raise ValueError("The number of basis functions should be the same as the length of the array x0.")   
    else:
        raise TypeError("The Gaussian centers, x0, parameter should be a float or an array.")
    
    # check argument xA
    if isinstance(xA, float):
        xA = np.full(n_sites, xA)
    elif isinstance(xA, np.ndarray):
        if xA.size != n_sites:
            raise ValueError("The number of sites should be the same as the length of the array xA.")   
    else:
        raise TypeError("The potential sites, xA, parameter should be a float or an array.")
    
    # check argument DA
    if isinstance(DA, float):
        DA = np.full(n_sites, DA)
    elif isinstance(DA, np.ndarray):
        if DA.size != n_sites:
            raise ValueError("The number of sites should be the same as the length of the array DA.")   
    else:
        raise TypeError("The potential depth, DA, parameter should be a float or an array.")
    
    # check argument aA
    if isinstance(aA, float):
        aA = np.full(n_sites, aA)
    elif isinstance(aA, np.ndarray):
        if aA.size != n_sites:
            raise ValueError("The number of sites should be the same as the length of the array aA.")   
    else:
        raise TypeError("The potential width, aA, parameter should be a float or an array.")

    # initialize V to a zero matrix
    V = np.zeros((n_basis,n_basis))
    
    for n in range(n_basis):
        pnn = alpha[n]**2
        for A in range(n_sites):
            pnA = alpha[n]*aA[A]
            snnA = 2*alpha[n]+aA[A]
            V[n,n] -= DA[A]*(4*pnn/snnA**2)**(1/4)*np.exp(-2*pnA/snnA*(x0[n]-xA[A])**2)
        for m in range(n):
            pmn =  alpha[m]*alpha[n]
            for A in range(n_sites):
                pmA = alpha[m]*aA[A]
                pnA = alpha[n]*aA[A]
                smnA = alpha[m]+alpha[n]+aA[A]
                V[m,n] -= DA[A]*(4*pmn/smnA**2)**(1/4)*(np.exp(-1*pmA/smnA*(x0[m]-xA[A])**2)
                                            *np.exp(-1*pnA/smnA*(x0[n]-xA[A])**2)
                                            *np.exp(-1*pmn/smnA*(x0[m]-x0[n])**2))
            V[n,m] = V[m,n]
    
    return V

def compute_V_numer(x0, alpha, n_basis,DA, aA, xA, n_sites):
    """Compute the potential energy <m|V|n> for 1-dimensional Gaussian wavefunctions by quadrature.

    V = sum(0,1,2,..n_sites) -DA exp(-aA*(x-xA)**2)
    
    Parameters
    ----------
    x0 : float or ndarray of size n_basis
        Position of the Gaussian center. All wells are
        assumed to be co-located if this is a float.
    alpha : float or ndarray of size n_basis
        exponential parameter in the normalized Gaussian wavefunction.
        If float; all Gaussians are assumed to have the same exponent.
    n_basis : scalar, int
        the number of Gaussian wavefunctions to be considered.
    DA : float or ndarray of size n_sites
        Depth of the potential well. All wells are co-located if this
        is a float.
    xA : float or ndarray of size n_sites
        Position of the potential well. All wells are co-located if this
        is a float.
    aA : float or ndarray of size n_sites
        exponential parameter controlling the width of the Gaussian well.
        If float; all wells have the same width. The half-width at half-maximum
        is sqrt((ln 2)/a) and the zero-point energy in the harmonic oscillator
        approximation is 1/2 sqrt(D * a).
    n_sites : scalar, int
        the number sites in the multi-potential well.   

    Returns
    -------
    V : array-like (nbasis, nbasis)
         Pot. E. matrix elements V_mn = <m | V | n> for the Gaussian wavefunctions.
         
    """ 
     # check argument n_basis
    if n_basis <= 0:
        raise ValueError("The number of basis functions should be positive.")
    
    # check argument n_sites
    if n_sites <= 0:
        raise ValueError("The number of sites should be positive.")
    
    # check argument alpha
    if isinstance(alpha, float):
        alpha = np.full(n_basis, alpha)
    elif isinstance(alpha, np.ndarray):
        if alpha.size != n_basis:
            raise ValueError("The number of basis functions should be the same as the length of the array alpha.")
    else:
        raise TypeError("The Gaussian width parameter should be a float or an array.")
    
    # check argument x0
    if isinstance(x0, float):
        x0 = np.full(n_basis, x0)
    elif isinstance(x0, np.ndarray):
        if x0.size != n_basis:
            raise ValueError("The number of basis functions should be the same as the length of the array x0.")   
    else:
        raise TypeError("The Gaussian centers, x0, parameter should be a float or an array.")
    
    # check argument xA
    if isinstance(xA, float):
        xA = np.full(n_sites, xA)
    elif isinstance(xA, np.ndarray):
        if xA.size != n_sites:
            raise ValueError("The number of sites should be the same as the length of the array xA.")   
    else:
        raise TypeError("The potential sites, xA, parameter should be a float or an array.")
    
    # check argument DA
    if isinstance(DA, float):
        DA = np.full(n_sites, DA)
    elif isinstance(DA, np.ndarray):
        if DA.size != n_sites:
            raise ValueError("The number of sites should be the same as the length of the array DA.")   
    else:
        raise TypeError("The potential depth, DA, parameter should be a float or an array.")
    
    # check argument aA
    if isinstance(aA, float):
        aA = np.full(n_sites, aA)
    elif isinstance(aA, np.ndarray):
        if aA.size != n_sites:
            raise ValueError("The number of sites should be the same as the length of the array aA.")   
    else:
        raise TypeError("The potential width, aA, parameter should be a float or an array.")

    # define function for potential
    def Vfun(x):
        V = 0
        for A in range(n_sites):
            V -= DA[A]*np.exp(-aA[A]*(x-xA[A])**2)
        return V

    # initialize V to a zero matrix
    V = np.zeros((n_basis,n_basis))
    
    for n in range(n_basis):
        integrand = lambda x: psi(x, x0[n], alpha[n])**2*Vfun(x)
        integral, error = quad(integrand,-np.inf,np.inf)
        V[n,n] = integral
        for m in range(n):
            integrand = lambda x: psi(x, x0[m], alpha[m])*psi(x, x0[n], alpha[n])*Vfun(x)
            integral, error = quad(integrand,-np.inf,np.inf)
            V[m,n] = integral
            V[n,m] = V[m,n]

    return V

# Check overlap integrals:
print("Overlap matrix computed analytically:", compute_S(np.array([1.0,2.0]), np.array([0.5,2.0]), 2))
print("Overlap matrix computed numerically:", compute_S_numer(np.array([1.0,2.0]), np.array([0.5,2.0]), 2))
print("Overlap matrix computed analytically:", compute_S(np.array([0,2.0]), 4.0, 2))
print("Overlap matrix computed numerically:", compute_S_numer(np.array([0,2.0]), 4.0, 2))

# Check kinetic energy integrals:
print("K.E. matrix computed analytically:", compute_T(np.array([1.0,2.0]), np.array([0.5,2.0]), 2))
print("K.E. computed numerically:", compute_T_numer(np.array([1.0,2.0]), np.array([0.5,2.0]), 2))
print("K.E. computed analytically:", compute_T(np.array([0,2.0]), 4.0, 2))
print("K.E. computed numerically:", compute_T_numer(np.array([0,2.0]), 4.0, 2))


# Check potential energy integrals:
print("Pot. E. matrix computed analytically:", compute_V(np.array([1.0,2.0]), np.array([0.5,2.0]), 2,np.array([2.0,4.0]),np.array([0.5,1.0]),np.array([0.0,2.0]),2))
print("Pot. E. computed numerically:", compute_V_numer(np.array([1.0,2.0]), np.array([0.5,2.0]), 2,np.array([2.0,4.0]),np.array([0.5,1.0]),np.array([0.0,2.0]),2))
print("Pot. E. computed analytically:", compute_V(np.array([0,2.0]), 4.0, 2, 4.0, 2.0, np.array([0.0,2.0]), 2))
print("Pot. E. computed numerically:", compute_V_numer(np.array([0,2.0]), 4.0, 2, 4.0, 2.0, np.array([0.0,2.0]), 2))


# ![MO diagram](MOdiagram.png 'MO diagram for 6 Gaussian wells')
# 
# ## &#x1f469;&#x1f3fd;&#x200d;&#x1f4bb; **Exercise 2.** Approximate treament of N electrons in a Gaussian multi-well.
# The Gaussian multiwell is a reasonable, albeit not perfect, model for a conjugated $\pi$ system or for a chain of alkali metal atoms. Obviously one needs different parameters for different systems, and in general both $D$ and $a$ will be smaller for Group 1 elements than for $\pi$ systems. We can choose, for simplicity, $D=1$, $a=0.5$, and the separation between adjacent sites as $R = 1$. 
# 
# ### &#x1f469;&#x1f3fd;&#x200d;&#x1f4bb; **Exercise 2a.** Write a function that computes the "molecular" orbitals and orbital energies for the multi-well potential with $P$ sites at $x_A = (A-1)$, $D_A = 2$, and $a_A = 4$. Choose the exponent for the Gaussian basis functions to be $\alpha = \sqrt{2}$, with one basis function per site. For 6 sites, your orbitals should look like those in the above figure.
# 
# ### &#x1f469;&#x1f3fd;&#x200d;&#x1f4bb; **Exercise 2b.** Write a function that approximates the total ground-state energy for $N$ electrons for the below Hamiltonian using the orbital energies as input.
# $$
# \hat{H}_{\text{multi}} = \sum_{i=0}^{N-1} \left(-\frac{1}{2} \frac{d^2}{dx_i^2} - \sum_{A=0}^{P-1} D_A \cdot e^{-a_A(x-x_A)^2} \right)
# $$
# The function will actually be general for any case with a potential like:
# $$
# \hat{H}_{\text{separable}} = \sum_{i=0}^{N-1} \left(-\frac{1}{2} \frac{d^2}{dx_i^2} + V(x_i) \right)
# $$
# where
# $$
# \left(-\frac{1}{2} \frac{d^2}{dx^2} + V(x) \right) \phi_k(x) = \epsilon_k \phi_k(x)
# $$

# ### Answer to Exercise 2a
# One needs to evaluate the secular equation, which consists of evaluating the generalized eigenproblem $(\mathbf{T}+\mathbf{V})\cdot \mathbf{c_k} = \epsilon_k \mathbf{S}\cdot \mathbf{c_k}$. The coefficients, $\mathbf{c_k}$ can be used to plot the orbitals.
# 

# In[7]:


# Exercise 2a
def compute_MOs(n_sites, alpha, D, a, R):
    """
    Compute the molecular orbitals and orbital energies for n_site Gaussian wells.
    
    Parameters
    ----------
    n_sites : int
        The number of sites, equal to the number of basis functions.
    alpha : float
        The Gaussian width parameter.
    R : float
        The separation between adjacent sites.
    D : float 
        The potential depth parameter.
    a : float 
        The potential width parameter.
    
    Returns
    -------
    epsilon: ndarray
        The orbital energies.
    orbs : ndarray
         Molecular orbital coefficients. Every column is a different orbital.
         
    """ 
    # check argument n_basis
    if n_sites <= 0:
        raise ValueError("The number of sites, n_sites, should be positive.")
    
    # check argument alpha
    if not (isinstance(alpha, float) and alpha > 0):
        raise TypeError("The Gaussian width parameter, alpha, should be a positive real number.")

    # check argument D
    if not (isinstance(D, float) and D > 0):
        raise TypeError("The well-depth parameter, D, should be a positive real number.")

    # check argument a
    if not (isinstance(a, float) and a > 0):
        raise TypeError("The well-width parameter, a, should be a positive real number.")

    # check argument R
    if isinstance(R, float):
        #Set up basis function/site centers to be equidistant
        x0 = np.zeros(n_sites)
        for i in range(n_sites):
            x0[i] = R*i 
    else:
        raise TypeError("The separation between adjacent sites, R, should be a float.")
    
    # Compute the molecular orbitals and orbital energies. Hint: use scipy.linalg.eigh
    ### START YOUR CODE HERE ###
    #Form the Hamiltonian matrix
    H = np.zeros((n_sites,n_sites))
    S = np.zeros((n_sites,n_sites))
    H = compute_T(x0,alpha,n_sites)+compute_V(x0,alpha,n_sites,D,a,x0,n_sites)
    S = compute_S(x0,alpha,n_sites)

    epsilon, orbs = scipy.linalg.eigh(H,S)

    return epsilon, orbs

    ### END YOUR CODE HERE ###

# Compute the molecular orbitals for a case of interest
epsilon, orbs = compute_MOs(6,np.sqrt(2.0),2.0,4.0,1.0)


# In[57]:


np.random.seed(42)
n_sites = 5
alpha = float(np.random.randint(1, 10))
D = np.random.randint(100, size=10) * 1e-2
a = np.random.uniform(size=10)
R = float(np.random.randint(1, 6))

answer_e = []
answer_mo = []
for D_, a_ in zip(D, a):
    e, mo = compute_MOs(n_sites,
                        alpha,
                        D_,
                        a_,
                        R)
    answer_e.append(e)
    answer_mo.append(mo)
    
answer_e = np.array(answer_e)

answer_e


# In[8]:


# Exercise 2b.
# Given orbital energies, compute the energy, assuming that electrons do not interact.
def compute_total_energy(N, epsilon):
    """
    Compute the total energy of a molecule with N electrons from its orbital energies,
    neglecting electron-electron repulsion. (I.e., the N-electron Hamiltonian is assumed
    to be a sum of atomic Hamiltonians.)
    
    Parameters
    ----------
    N : int
        The number of electrons.
    epsilon : ndarray
        The orbital energies.
    
    Returns
    -------
    E : float
        The total energy of the molecule.
    
    """
    # check argument N
    if not (isinstance(N, int) and N > 0):
        raise TypeError("The number of electrons, N, should be a positive integer.")
    
    # check argument epsilon
    if not (isinstance(epsilon, np.ndarray) and epsilon.size > N/2):
        raise TypeError("The orbital energies, epsilon, should be a one-dimensional array of at least size N/2.")
    
    # Compute the total ground-state energy from the orbital energies. 
    ### START YOUR CODE HERE ###
    # If the number of electrons is even, we fill up the orbitals up to N/2. 
    E = 0.0
    for i in range(N//2):
        E += 2*epsilon[i]
    #If the number of electrons is odd, we need to add one electron to the floor(N/2)+1 orbital.
    if N%2 == 1:
        E += epsilon[N//2]

    return E
    ### END YOUR CODE HERE ###
    

    
    
print("The orbital energies are:", epsilon)
print("The total energy for 5 electrons on a 6-site chain is:", compute_total_energy(5, epsilon))
print("The total energy for 6 electrons on a 6-site chain is:", compute_total_energy(6, epsilon))
print("The total energy for 7 electrons on a 6-site chain is:", compute_total_energy(7, epsilon))


# In[66]:


np.random.seed(42)
n_sites = 5
epsilon = np.random.rand(10, 10) * 1e-2

answer = []
for epsilon_ in epsilon:
    answer.append(compute_total_energy(n_sites, epsilon_))

answer


# In[9]:


import matplotlib.pyplot as plt


def calc_y_orbs_numerical(coeffs, alpha, D, a, R, xrange):
    n_sites = coeffs.shape[0]
    x0 = np.zeros(n_sites)
    for i in range(n_sites):
        x0[i] = R*i 
    
    y = 0
    for coeff, x_a in zip(coeffs, x0):
        y += coeff*((2*alpha/np.pi)**(1/4))*np.exp(-alpha*(xrange-x_a)**2)
    return y


def plot_orbitals(n_sites, alpha, D, a, R, x_left, x_right):
    """
    Plot the molecular orbitals and orbital energies for n_site Gaussian wells.
    
    Parameters
    ----------
    n_sites : int
        The number of sites, equal to the number of basis functions.
    alpha : float
        The Gaussian width parameter.
    R : float
        The separation between adjacent sites.
    D : float 
        The potential depth parameter.
    a : float 
        The potential width parameter.
    x_left: float
        The left boundary of the plot
    x_right: float
        The right boundary of the plot
    
    Returns
    -------
    None
    """
    epsilon, orbs = compute_MOs(n_sites, alpha, D, a, R)
    xrange = np.linspace(x_left, x_right, 1000)
    k = 0.5
    
    plt.figure(dpi=150)
    
    y_pot = 0
    
    max_y = -np.inf
    min_y = np.inf
    for i in range(n_sites):
        y_pot += -D*np.exp(-alpha*(xrange-R*i)**2)
        
    
    for i in range(n_sites):
        y_orb = k*calc_y_orbs_numerical(orbs[:, i], alpha, D, a, R, xrange)
        y_orb += epsilon[i]
        plt.plot(xrange, y_orb, label=f'$\psi_{i}$')
        plt.fill_between(xrange,
                         np.minimum(y_orb, epsilon[i]),
                         np.maximum(y_orb, epsilon[i]), alpha=0.5)
        
        plt.text(x_right-0.5, epsilon[i]+0.07, f'$E_{i}$')
        
        if np.max(y_orb)>max_y:
            max_y = np.max(y_orb)
        if np.min(y_orb)<min_y:
            min_y = np.min(y_orb)
    
    scale = (max_y - min_y) / (np.max(y_pot) - np.min(y_pot))
    y_pot *= scale*1.1
    
    plt.plot(xrange, y_pot+(max_y-np.max(y_pot)), 
             ls='--', c='black', lw=1, alpha=0.3)
        
    plt.legend(loc='upper left', framealpha=1)
    plt.show()
    
    
x_left, x_right = -3, 9
plot_orbitals(6, 5.0, 2.0, 4.0, 1.0, x_left, x_right)


# 
# 
# 
# ### &#x1f4b0; **BONUS:** [Peierls Distortion](https://en.wikipedia.org/wiki/Peierls_transition)
# It is well-known that long conjugated carbon chains evince bond-length alternation, even though (due to conjugation) technically all the bonds are (very nearly) equivalent. This is because of Peierls distortion. To understand the Peierls distortion, consider a model where odd-numbered bonds are shortened, but even-numbered bonds are lengthened. I.e.,
# 
# $$
# x_0 = 0.0
# x_A =  A*R + (-1)^{A+1}\xi  \qquad \qquad A=0,1,2,\ldots P-1
# $$
# 
# Here $0 < \xi \ll 1$ is a small number. This system is subject to a Peierls distortion if the energy decreases when $\xi$ is increased from zero, i.e., if $\tfrac{dE}{d\xi} < 0$. Note that it is important to choose the number of sites to be odd, as otherwise the above formula changes the length of the chain.
# 
# **Show that for sufficiently large odd $P$, the energy for $0 < \xi \ll 1$ is lower than the energy for $\xi=0$.** 
# 
# In fact, the effect even holds for sufficiently large *even* P, but you need to consider much larger values of $P$, and play with the key parameters a little bit, in order to see this.

# In[10]:


# Bonus:  test the Peierls distortion. 
### START YOUR CODE HERE ###
def compute_Peierls_MOs(n_sites, alpha, D, a, R, xi):
    """
    Compute the molecular orbitals and orbital energies for n_site Gaussian wells.
    
    Parameters
    ----------
    n_sites : int
        The number of sites, equal to the number of basis functions.
    alpha : float
        The Gaussian width parameter.
    R : float
        The separation between adjacent sites.
    D : float 
        The potential depth parameter.
    a : float 
        The potential width parameter.
    xi : float
        The bond-length alternation parameter.
    
    Returns
    -------
    epsilon: ndarray
        The orbital energies.
    orbs : ndarray
         Molecular orbital coefficients. Every column is a different orbital.
         
    """ 
    # check argument n_basis
    if n_sites <= 0:
        raise ValueError("The number of sites, n_sites, should be positive.")
    
    # check argument alpha
    if not (isinstance(alpha, float) and alpha > 0):
        raise TypeError("The Gaussian width parameter, alpha, should be a positive real number.")

    # check argument D
    if not (isinstance(D, float) and D > 0):
        raise TypeError("The well-depth parameter, D, should be a positive real number.")

    # check argument a
    if not (isinstance(a, float) and a > 0):
        raise TypeError("The well-width parameter, a, should be a positive real number.")
    
    # check argument xi
    if not (isinstance(xi, float)):
        raise TypeError("The bond-length alternation parameter, xi, should be a (small) real number.")

    # check argument R
    if isinstance(R, float):
        #Set up basis function/site centers to be equidistant
        x0 = np.zeros(n_sites)
        for i in range(n_sites):
            x0[i] = R*i + (-1)**(i+1)*xi
    else:
        raise TypeError("The separation between adjacent sites, R, should be a float.")
    
    # Compute the molecular orbitals and orbital energies. Hint: use scipy.linalg.eigh
    ### START YOUR CODE HERE ###
    #Form the Hamiltonian matrix
    H = np.zeros((n_sites,n_sites))
    S = np.zeros((n_sites,n_sites))
    H = compute_T(x0,alpha,n_sites)+compute_V(x0,alpha,n_sites,D,a,x0,n_sites)
    S = compute_S(x0,alpha,n_sites)

    epsilon, orbs = scipy.linalg.eigh(H,S)

    return epsilon, orbs

# Now we compute the orbitals and the corresponding energies for xi=0 and xi=small:



#These examples show both the even-P and odd-P case:
epsilon0, orbs0 = compute_Peierls_MOs(50,np.sqrt(2.0),2.0,4.0,3.0,0.0)
epsilon1, orbs1 = compute_Peierls_MOs(50,np.sqrt(2.0),2.0,4.0,3.0,0.01)
epsilon2, orbs2 = compute_Peierls_MOs(49,np.sqrt(2.0),2.0,4.0,3.0,0.0)
epsilon3, orbs3 = compute_Peierls_MOs(49,np.sqrt(2.0),2.0,4.0,3.0,0.01)

E0 = compute_total_energy(50, epsilon0)
E1 = compute_total_energy(50, epsilon1)
E2 = compute_total_energy(49, epsilon2)
E3 = compute_total_energy(49, epsilon3)

print("The total energy for xi=0 and P = 50 is:", E0)
print("The total energy for xi=0.01 and P = 50 is:", E1)
print("The total energy for xi=0 and P = 49 is:", E2)
print("The total energy for xi=0.01 and P = 49 is:", E3)

### END YOUR CODE HERE ###


# In[77]:


np.random.seed(42)
n_sites = 5
alpha = float(np.random.randint(1, 10))
D = np.random.randint(100, size=10) * 1e-2
a = np.random.uniform(size=10)
R = float(np.random.randint(1, 6))
xi = float(np.random.randint(1, 10))

answer_e = []
for D_, a_ in zip(D, a):
    epsilon, orbs = compute_Peierls_MOs(n_sites,
                                        alpha,
                                        D_,
                                        a_,
                                        R,
                                        xi)
    answer_e.append(epsilon)
    
answer_e = np.array(answer_e)
answer_e


# ## &#x2696;&#xfe0f; Marking Scheme
# &#x2611;&#xfe0f; Successful completion of the notebook, together with the ability to discuss your strategy, earns an **S**.  If you complete Exercise 1 *or* Exercise 2, and discuss your solution, then you can earn an S-.  
# 
# &#x1f4b0; For an **S+**, Complete the bonus on the Peierls Distortion.

# 
