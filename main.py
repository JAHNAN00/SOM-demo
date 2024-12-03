import numpy as np
import random
import matplotlib.pyplot as plt


class SOM(object):
    def __init__(self,net=[[1,1],[1,1]],epochs = 500,r_t = [None,None],eps=1e-6):
        self.epochs = epochs
        self.C = r_t[0]
        self.B = r_t[1]
        self.eps = eps
        self.output_net = np.array(net)
        if len(self.output_net.shape) == 1:
            self.output_net = self.output_net.reshape([-1,1])
        self.coord = np.zeros([self.output_net.shape[0],self.output_net.shape[1],2])
        for i in range(self.output_net.shape[0]):
            for j in range(self.output_net.shape[1]):
                self.coord[i,j] = [i,j]


    def __r_t(self,t):
        if not self.C:
            return 0.5
        else:
            return self.C*np.exp(-self.B*t/self.epochs)

    def __lr(self,t,distance):
        return (self.epochs-t)/self.epochs*np.exp(-distance)
    def standard_x(self,x):
        x = np.array(x)
        for i in range(x.shape[0]):
            x[i,:] = [value/(((x[i,:])**2).sum()**0.5) for value in x[i,:]]
        return x
    def standard_w(self,w):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                w[i,j,:] = [value/(((w[i,j,:])**2).sum()**0.5) for value in w[i,j,:]]
        return w
    def cal_similar(self,x,w):
        similar = (x*w).sum(axis=2)
        coord = np.where(similar==similar.max())
        return [coord[0][0],coord[1][0]]

    def update_w(self,center_coord,x,step):
        for i in range(self.coord.shape[0]):
            for j in range(self.coord.shape[1]):
                distance = (((center_coord-self.coord[i,j])**2).sum())**0.5
                if distance <= self.__r_t(step):
                    self.W[i,j] = self.W[i,j] + self.__lr(step,distance)*(x-self.W[i,j])

    def transform_fit(self,x):
        self.train_x = self.standard_x(x)
        self.W = np.zeros([self.output_net.shape[0],self.output_net.shape[1],self.train_x.shape[1]])
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                self.W[i,j,:] = self.train_x[random.choice(range(self.train_x.shape[0])),:]
        self.W = self.standard_w(self.W)
        for step in range(int(self.epochs)):
            j = 0
            if self.__lr(step,0) <= self.eps:
                break
            for index in range(self.train_x.shape[0]):
                center_coord = self.cal_similar(self.train_x[index,:],self.W)
                self.update_w(center_coord,self.train_x[index,:],step)
                self.W = self.standard_w(self.W)
                j += 1
        label = []
        for index in range(self.train_x.shape[0]):
            center_coord = self.cal_similar(self.train_x[index, :], self.W)
            label.append(center_coord[1]*self.coord.shape[1] + center_coord[0])
        class_dict = {}
        for index in range(self.train_x.shape[0]):
            if label[index] in class_dict.keys():
                class_dict[label[index]].append(index)
            else:
                class_dict[label[index]] = [index]
        cluster_center = {}
        for key,value in class_dict.items():
            cluster_center[key] = np.array([x[i, :] for i in value]).mean(axis=0)
        self.cluster_center = cluster_center

        return label


    def fit(self,x):
        self.train_x = self.standard_x(x)
        self.W = np.random.rand(self.output_net.shape[0], self.output_net.shape[1], self.train_x.shape[1])
        self.W = self.standard_w(self.W)
        for step in range(int(self.epochs)):
            j = 0
            if self.__lr(step,0) <= self.eps:
                break
            for index in range(self.train_x.shape[0]):
                center_coord = self.cal_similar(self.train_x[index, :], self.W)
                self.update_w(center_coord, self.train_x[index, :], step)
                self.W = self.standard_w(self.W)
                j += 1
        label = []
        for index in range(self.train_x.shape[0]):
            center_coord = self.cal_similar(self.train_x[index, :], self.W)
            label.append(center_coord[1] * self.coord.shape[1] + center_coord[1])
        class_dict = {}
        for index in range(self.train_x.shape[0]):
            if label[index] in class_dict.keys():
                class_dict[label[index]].append(index)
            else:
                class_dict[label[index]] = [index]
        cluster_center = {}
        for key, value in class_dict.items():
            cluster_center[key] = np.array([x[i, :] for i in value]).mean(axis=0)
        self.cluster_center = cluster_center

    def predict(self,x):
        self.pre_x = self.standard_x(x)
        label = []
        for index in range(self.pre_x.shape[0]):
            center_coord = self.cal_similar(self.pre_x[index, :], self.W)
            label.append(center_coord[1] * self.coord.shape[1] + center_coord[1])
        return label

if __name__ == '__main__':
    som = SOM(epochs=50)
    data_class_1 = np.random.multivariate_normal([5, -5], [[1, 0.5], [0.5, 1]], 200)
    data_class_2 = np.random.multivariate_normal([-2, -2], [[1, -0.5], [-0.5, 1]], 200)
    data_class_3 = np.random.multivariate_normal([2, 2], [[1, 0.5], [0.5, 1]], 200)
    data_class_4 = np.random.multivariate_normal([-5, 7], [[1, -0.5], [-0.5, 1]], 200)
    
    data = np.vstack((data_class_1, data_class_2,data_class_3,data_class_4))
    labels = np.array([0] * 200 + [1] * 200)
   
    x = data
    y_pre = som.transform_fit(x)
    colors = "rgby"
    figure = plt.figure(figsize=[20,12])
    plt.scatter(x[:,0],x[:,1],c=[colors[i] for i in y_pre])
    plt.savefig('main.png')
