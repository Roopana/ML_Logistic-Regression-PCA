# Logistic Regression & PCA
- Compare my 2 Class Logistic Regression with Scikit's Logistic Regression using Boston housing dataset
- Perform PCA using Mixture of Gaussian models 

## Logistic Regression 
Build a 2 class Logistic Regression classification model without using sci-kit library functions. Develop code with a set of parameters (w,w0) where w ∈ Rd, w0 ∈ R. Assuming the two classes are {0,1}, and the data x ∈ Rd, the posterior probability of class C1 is given by

__P(1|x)= exp(wTx+w0) / 1+exp(wTx+w0)__ \
and __P(0|x) = 1 − P(1|x)__

## PCA
Compute a PCA projection Z ∈ Rd×n, d ≤ D of the original data X ∈ RD×n so that α% of the variance is preserved in the projected space. Develop code only using numpy and scipy libraries. 

The feature covariance matrix of the data can be computed as:

__Σ=1/n Σ(xj−μˆ)(xj−μˆ)T__

where __μˆ = n1 \
􏰉nj=1 \
xj__ is the mean of the data points
