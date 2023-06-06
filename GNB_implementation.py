#Gaussian Naive Bayes tutorial on a data set of 20 points
#with 3 classes and 2 distinct features
#credits to Dr Robert Kubler
#https://towardsdatascience.com/learning-by-implementing-gaussian-naive-bayes-3f0e3d2c01b2
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
import numpy as np

#create the dataset using sklearn.datasets' function "make_blobs"
#n_samples = # of data points, centers = create different classes
x, y = make_blobs(n_samples=20, centers=[(0,0), (5,5), (-5,5)], random_state=0)

#priors for each class 
#total occurence of each class / total # of data points
#np.bincounts counts the occurence of each label
prior = np.bincount(y) / len(y)

#np.where(y==i) returns all indices where y==i
#mean and standard deviations of the data points of each class using numpy's mean and standard deviation functions
mean = np.array([x[np.where(y==i)].mean(axis = 0) for i in range (3)])
stds = np.array([x[np.where(y==i)].std(axis=0) for i in range(3)])

print(prior)
print(mean)
print(stds)

#a random point at point coordinate (-2,5)
x_new = np.array([-2,5])

#probability of the class that x_new is belong to 
#using the unnormalised Gaussian Naive Bayes formula 
#ignoring p(x) (p(data))
for j in range(3):
    print(f'Unnormalised probability of x_new belongs to class {j}: {(1/np.sqrt(2*np.pi*stds[j]**2)*np.exp(-0.5*((x_new-mean[j])/stds[j])**2)).prod()*prior[j]:.12f}')

#divide these unnormalised probability to their sum of 0.00032569 yields
for j in range(3):
    print(f'Normalised probability of x_new belongs to class {j}: {((1/np.sqrt(2*np.pi*stds[j]**2)*np.exp(-0.5*((x_new-mean[j])/stds[j])**2)).prod()*prior[j])/0.00032569:.12f}')

#using sklearn GaussianNB library 
gnb = GaussianNB()
gnb.fit(x,y)
print(gnb.predict_proba([[-2,5]]))

