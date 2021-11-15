import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap


#Q6
def gen_h(num):
    x = np.random.rand(num, 2)              #x is the data point with 2 dim
    y = np.random.randint(2, size=num)      #y is the label
    return x, y


'''
from sklearn.neighbors import KNeighborsClassifier as KNN
def KNN_sk(test_pt, model_x, model_y, k):      #KNN from sklearn for testing
    neigh = KNN(n_neighbors = k)
    neigh.fit(model_x, model_y)
    y_test = neigh.predict(test_pt)
    return y_test
'''

def knn(test_pt, model_x, model_y, k):      #knn alg
    test_y = [None] * len(test_pt)          #create empty list for test_pt labels
    for i in range (len(test_pt)):
        distance = list(range(2, k+2))      #create k initiate distance with min=2
        test_yi = [None] * k                #create empty list for k labels
        for j in range (len(model_x)):
            max_value = max(distance)
            max_index = distance.index(max_value)    #find max value and index in distance
            dis = np.linalg.norm(test_pt[i] - model_x[j])
            if dis < max_value:
                distance[max_index] = dis   #update the distance
                test_yi[max_index] = model_y[j]      #update associated label
        test_y[i] = decide_y(test_yi, k)
    return test_y


def decide_y(y,k):
    if sum(y) < k/2:
        return 0
    elif sum(y) == k/2:
        return np.random.randint(2)
    else:
        return 1
            

#Q7
def gen_b(num, model_x, model_y):
    x_b = np.random.rand(num, 2)      #x is the data point with 2 dim
    y_b = np.zeros(num)
    y_b_knn = knn(x_b, model_x, model_y, 3)  #label obtained from knn hsv v=3
    for i in range (num):
        if random.random() < 0.2:           #Tail come up
            y_b[i] = np.random.randint(2)
        else:                               #Head come up
            y_b[i] = y_b_knn[i]             #y_b label from knn
    return x_b, y_b


def mse(y_knn, y_gen):                      #generalisation error
    len(y_knn) == len(y_gen)                #check y_knn, y_gen same size
    sse = 0                                 #square error
    for i in range (len(y_gen)):
        sse += np.square(y_knn[i] - y_gen[i])
    return sse/len(y_gen)
        
            
   
def prot_a(k):                              #Protocal A
    gen_err = []
    for i in range(100):
        x, y = gen_h(100)                    #sample h
        x_b, y_b = gen_b(4000, x, y)        #generate m training points
        x_gen, y_gen = gen_b(1000, x, y)    #generate test points
        y_knn = knn(x_gen, x_b, y_b, k)     #run knn on test points
        err = mse(y_knn, y_gen)
        gen_err.append(err)
    #print('k =', k, 'done :)')
    return sum(gen_err) / len(gen_err)      #mean of gen_err


def prot_b(m):                              #Protocal B
    opt_k=[]
    for i in range (100):
        gen_err = []
        for k in range (1,50):                  #k = 1 to 49
            x,y = gen_h(100)                    #sample h
            x_b, y_b = gen_b(m, x, y)           #generate m training points
            x_gen, y_gen = gen_b(1000, x ,y)    #generate test points
            y_knn = knn(x_gen, x_b, y_b, k)     #run knn on test points
            err = mse(y_knn, y_gen)
            gen_err.append(err)
        min_err = min(gen_err)                  #find min error over k
        k_min_err = gen_err.index(min_err) + 1  #return k with min_err
        opt_k.append(k_min_err)
        #print(i)
    return sum(opt_k) / len(opt_k)              #mean of optimal k
        


#Ans Q6
def ans_6():
    x, y = gen_h(100)
    h = .002
    cmap_a = ListedColormap(['#AAAAFF', '#AAFFAA'])
    cmap_b = ListedColormap(['#0000FF', '#00FF00'])
    x_min, x_max = 0, 1.02
    y_min, y_max = 0, 1.02
    #x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    #y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    m, n = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = knn(np.c_[m.ravel(), n.ravel()],x,y,3)
    Z = np.asarray(Z)
    # Put the result into a color plot
    Z = Z.reshape(m.shape)
    plt.figure()
    plt.pcolormesh(m, n, Z,cmap=cmap_a)
    plt.scatter(x[:, 0],x[:, 1],c=y, cmap=cmap_b)
    plt.show()    
    
#Ans Q7
def ans_7():
    k = list(range(1, 50))
    err = []
    for element in k:
        err.append(prot_a(element))
    plt.scatter(k, err)
    plt.xlabel('k')
    plt.ylabel('generalisation error')
    plt.show()
    return k, err

#Ans Q8
def ans_8():
    m = [100] + [500*x for x in range (1,9)]
    k = []
    for element in m:
        k.append(prot_b(element))
        print(element)
    plt.scatter(m,k)
    plt.yscale('log') 
    plt.xlabel('m')
    plt.ylabel('optimal k')
    plt.show()
    return m, k
    
    
ans_6()
ans_7()    
ans_8()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

