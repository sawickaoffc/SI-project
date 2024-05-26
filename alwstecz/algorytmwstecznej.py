import numpy as np
import pandas as pd
import nnet as net
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from timeit import default_timer as timer

class SonarClassifier:
    def __init__(self, x, y_t, K1, K2, lr, err_goal, disp_freq, mc, max_epoch, initialize):
        self.x = x
        self.L = self.x.shape[0]
        self.y_t = y_t
        self.K1 = K1
        self.K2 = K2
        self.lr = lr
        self.err_goal = err_goal
        self.disp_freq = disp_freq
        self.mc = mc
        self.max_epoch = max_epoch
        self.K3 = y_t.shape[0]
        self.SSE_vec = []
        self.PK_vec = []
        self.data = self.x.T
        self.target = self.y_t
        self.initialize = initialize
        
        if self.initialize:
            self.w1, self.b1 = net.nwtan(self.K1, self.L)
            self.w2, self.b2 = net.nwtan(self.K2, self.K1)
            self.w3, self.b3 = net.rands(self.K3, self.K2)
        else:
            self.w1, self.b1, self.w2, self.b2, self.w3, self.b3 = self.load_weights('sonar_weights.csv')

        self.w1_t_1, self.b1_t_1, self.w2_t_1, self.b2_t_1, self.w3_t_1, self.b3_t_1 = \
        self.w1, self.b1, self.w2, self.b2, self.w3, self.b3
        self.SSE = 0
        self.lr_vec = list()

    def load_weights(self, filename):
        weights = pd.read_csv(filename, header=None).values
        w1, b1 = weights[:self.K1], weights[self.K1:self.K1+1]
        w2, b2 = weights[self.K1+1:self.K1+1+self.K2], weights[self.K1+1+self.K2:self.K1+1+self.K2+1]
        w3, b3 = weights[self.K1+1+self.K2+1:], weights[self.K1+1+self.K2+1+self.K3:self.K1+1+self.K2+1+self.K3+1]
        return w1, b1, w2, b2, w3, b3

    def predict(self, x):
        n = np.dot(self.w1, x)
        self.y1 = net.tansig(n, self.b1 * np.ones(n.shape))
        n = np.dot(self.w2, self.y1)
        self.y2 = net.tansig(n, self.b2 * np.ones(n.shape))
        n = np.dot(self.w3, self.y2)
        self.y3 = net.purelin(n, self.b3 * np.ones(n.shape))
        return self.y3

    def train(self, x_train, y_train):
        for epoch in range(1, self.max_epoch + 1):
            self.y3 = self.predict(x_train)
            self.e = y_train - self.y3

            self.SSE_t_1 = self.SSE
            self.SSE = net.sumsqr(self.e)
            self.PK = (1 - sum((abs(self.e) >= 0.5).astype(int)[0]) / self.e.shape[1]) * 100
            self.PK_vec.append(self.PK)
            if self.SSE < self.err_goal or self.PK == 100:
                break

            if np.isnan(self.SSE):
                break

            self.d3 = net.deltalin(self.y3, self.e)
            self.d2 = net.deltatan(self.y2, self.d3, self.w3)
            self.d1 = net.deltatan(self.y1, self.d2, self.w2)
            self.dw1, self.db1 = net.learnbp(x_train, self.d1, self.lr)
            self.dw2, self.db2 = net.learnbp(self.y1, self.d2, self.lr)
            self.dw3, self.db3 = net.learnbp(self.y2, self.d3, self.lr)

            self.w1_temp, self.b1_temp, self.w2_temp, self.b2_temp, self.w3_temp, self.b3_temp = \
                self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy(), self.w3.copy(), self.b3.copy()

            self.w1 += self.dw1 + self.mc * (self.w1 - self.w1_t_1)
            self.b1 += self.db1 + self.mc * (self.b1 - self.b1_t_1)
            self.w2 += self.dw2 + self.mc * (self.w2 - self.w2_t_1)
            self.b2 += self.db2 + self.mc * (self.b2 - self.b2_t_1)
            self.w3 += self.dw3 + self.mc * (self.w3 - self.w3_t_1)
            self.b3 += self.db3 + self.mc * (self.b3 - self.b3_t_1)

            self.w1_t_1, self.b1_t_1, self.w2_t_1, self.b2_t_1, self.w3_t_1, self.b3_t_1 = \
                self.w1_temp, self.b1_temp, self.w2_temp, self.b2_temp, self.w3_temp, self.b3_temp

            self.SSE_vec.append(self.SSE)

    def train_CV(self, CV, skfold):
        PK_vec = np.zeros(CV)

        for i, (train, test) in enumerate(skfold.split(self.data, np.squeeze(self.target)), start=0):
            x_train, x_test = self.data[train], self.data[test]
            y_train, y_test = np.squeeze(self.target)[train], np.squeeze(self.target)[test]

            self.train(x_train.T, y_train.T)
            result = self.predict(x_test.T)

            n_test_samples = test.size
            PK_vec[i] = (1 - sum((abs(result - y_test) >= 0.5).astype(int)[0]) / n_test_samples) * 100

        PK = np.mean(PK_vec)
        return PK

#tutaj ogarniam baze
# Wczytanie danych sonarowych
data = pd.read_csv('sonar.all-data', header=None)
x = data.iloc[:, :-1].values.T
y_t = pd.get_dummies(data.iloc[:, -1]).values.T

print("Shape of x:", x.shape)
print("Shape of y_t:", y_t.shape)

print("Liczba próbek w danych wejściowych X:", x.shape[0])
print("Liczba próbek w etykietach y_one_hot:", y_t.shape[0])

# Parametry modelu
max_epoch = 1000
err_goal = 0.25
disp_freq = 10
lr_vec = np.array([1e-3, 1e-4, 1e-5])
mc_vec = np.array([0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
K1_vec = np.array([1, 3, 5, 7])
K2_vec = K1_vec

start = timer()
CVN = 10
skfold = StratifiedKFold(n_splits=CVN)

# Ewaluacja modelu
PK_2D_K1K2 = np.zeros([len(K1_vec), len(K2_vec)])
PK_2D_lrmc = np.zeros([len(lr_vec), len(mc_vec)])
PK_2D_K1K2_max = 0
k1_ind_max = 0
k2_ind_max = 0

# Ewaluacja parametrów K1 i K2
for k1_ind in range(len(K1_vec)):
    for k2_ind in range(len(K2_vec)):
        mlpnet = SonarClassifier(x, y_t, K1_vec[k1_ind], K2_vec[k2_ind], lr_vec[-1], err_goal, disp_freq, mc_vec[-1], max_epoch, True)
        PK = mlpnet.train_CV(CVN, skfold)
        print("K1 {} | K2 {} | PK {}".format(K1_vec[k1_ind], K2_vec[k2_ind], PK))
        PK_2D_K1K2[k1_ind, k2_ind] = PK
        if PK > PK_2D_K1K2_max:
            PK_2D_K1K2_max = PK
            k1_ind_max = k1_ind
            k2_ind_max = k2_ind

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(K1_vec, K2_vec)
surf = ax.plot_surface(X, Y, PK_2D_K1K2.T, cmap='viridis')
ax.set_xlabel('K1')
ax.set_ylabel('K2')
ax.set_zlabel('PK')
ax.view_init(30, 200)
plt.savefig("Fig.1_PK_K1K2_sonar.png", bbox_inches='tight')

PK_2D_lrmc_max = 0
lr_ind_max = 0
mc_ind_max = 0

# Ewaluacja parametrów lr i mc
for lr_ind in range(len(lr_vec)):
    for mc_ind in range(len(mc_vec)):
        mlpnet = SonarClassifier(x, y_t, K1_vec[k1_ind_max], K2_vec[k2_ind], lr_vec[lr_ind], err_goal, disp_freq, mc_vec[mc_ind], max_epoch, True)
        PK = mlpnet.train_CV(CVN, skfold)
        print("lr {} | mc {} | PK {}".format(lr_vec[lr_ind], mc_vec[mc_ind], PK))
        PK_2D_lrmc[lr_ind, mc_ind] = PK
        if PK > PK_2D_lrmc_max:
            PK_2D_lrmc_max = PK
            lr_ind_max = lr_ind
            mc_ind_max = mc_ind

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(lr_vec, mc_vec)
surf = ax.plot_surface(X, Y, PK_2D_lrmc.T, cmap='viridis')
ax.set_xlabel('lr')
ax.set_ylabel('mc')
ax.set_zlabel('PK')
ax.view_init(30, 200)
plt.savefig("Fig.2_PK_lrmc_sonar.png", bbox_inches='tight')

print("OPTYMALNE WARTOŚCI: K1={} | K2={} | lr={} | mc={} | PK={}".format(K1_vec[k1_ind_max], K2_vec[k2_ind_max], lr_vec[lr_ind_max], mc_vec[mc_ind_max], PK_2D_lrmc[lr_ind_max, mc_ind_max]))
print("Execution time:", timer() - start)
