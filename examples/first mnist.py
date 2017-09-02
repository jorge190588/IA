
# coding: utf-8

# In[51]:

import numpy as np
#image tools
from matplotlib import pyplot as plt
from PIL import Image
get_ipython().magic(u'matplotlib inline')

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[52]:

print("The training set is:")
print(mnist.train.images.shape)
print(mnist.train.labels.shape)

print("The Test set is:")
print(mnist.test.labels.shape)
print(mnist.test.labels.shape)

print("Validation set?")
print(mnist.validation.labels.shape)
print(mnist.validation.labels.shape)


# In[53]:

label = 5
mnist.train.images[label]


# In[54]:

mnist.train.images[label].shape


# In[55]:

a = np.reshape(mnist.train.images[label], [28,28])


# In[56]:

a.shape


# In[57]:

plt.imshow(a, cmap='Greys_r')

