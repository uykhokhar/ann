
# coding: utf-8

# In[118]:

import numpy as np
import random
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder
import pandas as pd


# In[119]:

def g_new(z):
    return 1.0/(1 + np.exp(-z))

def gprime(z):
    return np.multiply(g_new(z),(1-g_new(z)))


# In[131]:

def xavier(n, h, s, y=1): #num cols in X, hidden layers, nodes in each layer, num column in y 
    w = []
    n_in = 0
    n_out = 0
    for layer in range(h+1):
        if layer == 0: #first layer
            n_in = n
            n_out = s
        elif layer == h: #last layer
            n_in = s
            n_out = y
        else:
            n_in = s
            n_out = s
        xav = np.sqrt(6/(n_in + n_out))
        ran_w = np.random.random((n_in, n_out)) 
        ran_w = 2 * ran_w * xav - xav #mean 0, range -xav to xav
        w.append(ran_w)#append to matrices of weights
    return w

def xavier_bias(h,s,y=1):
    b = []
    n_in = s
    n_out = s
    for layer in range(h):
        xav = np.sqrt(6/(n_in + n_out))
        ran_w = np.random.random((1, n_out)) 
        ran_w = 2 * ran_w * xav - xav #mean 0, range -xav to xav
        b.append(ran_w)
    ran_w = np.random.random((1,y)) #for last y
    b.append(ran_w)
    return b

b = xavier_bias(2,4, 2)
w = xavier(3, 2,4, 2)
print(b)
print(w)
for item in b:
    print(item.shape)


# In[124]:

#X = np.array([[0,0,1], [0,1,1], [1,0,1],[1,1,1]])
                
#y = np.array([[0], [1], [1], [0]])

#Simple data set
X = np.array([  [0,0,1], [0,1,1],[1,0,1],[1,1,1] ])            
y = np.array([[0,0,1,1]]).T


t = 10000 #max iterations
alpha = 0.5
h = 2
s = 4

np.random.seed(1)

def ann_fit(X,y,alpha,t,h,s):
    #One-hot-encoder of y
    enc = OneHotEncoder()
    enc.fit(y)
    y = enc.transform(y).toarray()
    print(y.shape)
    
    
    # Xavier initialization
    w = xavier(X.shape[1], h, s, y=y.shape[1])
    b = xavier_bias(h,s, y=y.shape[1])

    print("w shape", w)
    print("b shape", b)

    for iteration in range(t):
        
        if (iteration % 1000 == 0):
            loss_all = []

        for ex in range(X.shape[0]):
            a_layer = []
            in_layer = []
            del_layer = [0]*(h+2)

            a_i = np.asmatrix(X[ex])
            a_layer.append(a_i)
            y_j = np.asmatrix(y[ex])

            for layer in range(0,h+1):
                a_i = a_layer[-1]
                in_cur = np.dot(a_i,w[layer]) + b[layer]
                a_j = g_new(in_cur)
                
                #save for later
                in_layer.append(in_cur)
                a_layer.append(a_j)

            #back propogation
            in_j = in_layer[-1]
            a_j = a_layer[-1]
            print("yj shape{} {}".format(y_j.shape, y_j))
            print("a_jshape{} {}".format(a_j.shape, a_j))
            loss = np.mean(np.asarray(y_j - a_j)**2)/2.0

            
            if (iteration % 1000 == 0):
                loss_all.append(loss)
                if ex == 0:
                    print("************ ITERATION {}**********".format(iteration))
                print("Ex {}, a_j output {}".format(ex, a_j))
                if ex == X.shape[0] -1:
                    mean_loss = np.mean(loss_all)
                    print("loss {}".format(mean_loss))
            print("gprime(in_j)", gprime(in_j))    
            
            
            del_j = np.multiply(gprime(in_j),(y_j - a_j))
            print("del_j shape", del_j.shape)
            
            del_layer[h+1] = del_j.T

            for layer in range(h, 0, -1):
                #print("*****layer {}".format(layer))
                #element wise multiplication of g'(in_i) * dot product of (w_ij , delta J)
                print("gprime(in_layer[layer])", gprime(in_layer[layer]).shape)
                print("in_layer[layer]", in_layer[layer].shape)
                print("w[layer]", w[layer].shape)
                print("del_layer[layer+1] shape", del_layer[layer+1].shape)
                print("dot_w_a", dot_w_a.shape)
                
                dot_w_a = np.dot(w[layer], del_layer[layer+1])
                del_i = np.multiply(gprime(in_layer[layer]).T, dot_w_a) 
                
                #print("np.dot(w[layer], del_layer[layer+1])", np.dot(w[layer], del_layer[layer+1]).shape)
                print("del_i shape {}".format(del_i.shape))
                
                del_layer[layer] = del_i

            for layer in range(h, -1, -1):
                #print("*****update w layer {}".format(layer))
                #print("a_layer[layer].T ", a_layer[layer].T.shape)
                #print("del_layer[layer + 1].T" , del_layer[layer + 1].T.shape)
                w[layer] += alpha * a_layer[layer].T * del_layer[layer + 1].T

                b[layer] += alpha * del_layer[layer + 1].T
    return w, b, enc


# In[125]:

class ANN: 
    
    def __init__(self,h,s):
        self.h = h
        self.s = s
        self.w = 0
        self.b = 0
        self.enc = 0
    
    def fit(self,X, y, alpha, t):
        self.w, self.b, self.enc = ann_fit(X,y,alpha,t,self.h,self.s)
  
    
    def predict(self,T):
        w = self.w
        b = self.b
        
        for ex in range(T.shape[0]):
            a_layer = []
            in_layer = []
            #del_layer = [0]*(h+2)

            a_i = np.asmatrix(T[ex])
            a_layer.append(a_i)
            #y_j = y[ex]

            for layer in range(0,h+1):
                a_i = a_layer[-1]
                #if iteration == 1:
                    #print("layer {}".format(layer))
                    #print("b_layer: ", b[layer].shape)
                    #print("np.dot(a_i,w[layer])", np.dot(a_i,w[layer]).shape)
                in_cur = np.dot(a_i,w[layer]) + b[layer]
                in_layer.append(in_cur)
                a_j = g_new(in_cur)
                a_layer.append(a_j)
            
        return a_layer[-1]       
        
    def print(self,):
        print(self.w)
    
        


# In[126]:

#X = np.array([[0,0,1], [0,1,1], [1,0,1],[1,1,1]])
                
#y = np.array([[0], [1], [1], [0]])

#Simple data set
X = np.array([  [0,0,1], [0,1,1],[1,0,1],[1,1,1] ])            
y = np.array([[0,0,1,1]]).T


t = 60000 #max iterations
alpha = 0.01
h = 2
s = 4

a1 = ANN(h,s)
a1.fit(X,y,alpha,t)
a1.print()

a_layer = a1.predict(X)
print("weight predict", w)
print("b predict", b)
print("a_layer", a_layer)


# In[25]:

#X = np.array([[0,0,1], [0,1,1], [1,0,1],[1,1,1]])
                
#y = np.array([[0], [1], [1], [0]])

#Simple data set
X = np.array([  [0,0,1], [0,1,1],[1,0,1],[1,1,1] ])            
y = np.array([[0,0,2,1]]).T

enc_label = LabelEncoder()
train_data = enc_label.fit_transform(y[:,0])

# do the others
for i in range(1, y.shape[1]):
    enc_label = LabelEncoder()
    train_data = np.column_stack((train_data, enc_label.fit_transform(y[:,i])))

train_categorical_values = train_data.astype(float)

enc = OneHotEncoder()
train_cat_data = enc.fit_transform(y)
#train_cat_data


enc = OneHotEncoder()
hy = enc.fit(y)
enc.transform([[0], [1], [1], [0]]).toarray()


# In[28]:

print(enc.n_values_)

print(enc.feature_indices_)

enc.transform([[0], [1], [1], [0]]).toarray()


# In[111]:

x = np.array([[1,2],[2,3],[3,4],[5,6]])
y = np.array([[0],[7]])
z = np.dot(x,y)
z


# In[ ]:



