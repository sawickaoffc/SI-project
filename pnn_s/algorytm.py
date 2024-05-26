import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import hickle as hkl
x,y_t,x_norm,x_n_s,y_t_s = hkl.load('iris.hkl')
# x,y_t,x_norm = hkl.load('wine.hkl')
if min(min(y_t))>0:
 y_t -= min(min(y_t))
x=x.T
y_t = np.squeeze(y_t)
data = x
target = y_t
CVN = 10
skfold = StratifiedKFold(n_splits=CVN)
PK_vec = np.zeros(CVN)
uniform = lambda x,b: (np.abs(x/b) <= 1) and 1/2 or 0
triangle = lambda x,b: (np.abs(x/b) <= 1) and (1 - np.abs(x/b)) or 0
gaussian = lambda x,b: (1.0/np.sqrt(2*np.pi))* np.exp(-.5*(x/b)**2)
laplacian = lambda x,b: (1.0/(2*b))* np.exp(-np.abs(x/b))
epanechnikov = lambda x,b: (np.abs(x/b)<=1) and ((3/4)*(1-(x/b)**2)) or 0
def pattern_layer(inp,kernel,sigma):
 k_values=[]
 for i,p in enumerate(x_train):
 edis = np.linalg.norm(p-inp) #find eucliden distance
 k = kernel(edis,sigma) #pass values of euclidean dist and
 #smoothing parameter to kernel function
 k_values.append(k)
 return k_values
def summation_layer(k_values,Y_train,class_uniques, class_counts):
 # Summing up each value for each class and then averaging
 summed = np.zeros(len(class_uniques))
 for i,c in enumerate(class_uniques):
 val = (Y_train==class_uniques[i])
 k_values = np.array(k_values)
 summed[i] = np.sum(k_values[val])
 avg_sum = list(summed/class_counts)
 return avg_sum
def output_layer(avg_sum,class_uniques):
 maxv = max(avg_sum)
 label = class_uniques[avg_sum.index(maxv)]
 return label
## Bringing all layers together under PNN function
def pnn(X_train,Y_train,X_test,kernel,sigma):
 # Initialising variables
 class_uniques = np.unique(Y_train)
 class_number = len(np.unique(Y_train))
 class_counts = np.zeros(class_number)
 for i,class_val in enumerate(class_uniques):
 class_counts[i] = sum(class_val==Y_train)
 labels=[]
 #Passing each sample observation
 for s in X_test:
 k_values = pattern_layer(s,kernel,sigma)
 avg_sum = summation_layer(k_values,Y_train,class_uniques, class_counts)
 label = output_layer(avg_sum,class_uniques)
 labels.append(label)
 # print('Labels Generated for bandwidth:',sigma)
 return labels
#Candidate Kernels
kernels_vec = ["Gaussian","Triangular","Epanechnikov", "Uniform", "Laplacian"]
18
sigmas_vec = np.arange(0.1, 2.2, 0.2) #[0.05,0.5,0.8,1,1.2]
PK_krn_sig_vec = np.zeros([len(kernels_vec),len(sigmas_vec)])
for kernels_ind in range(len(kernels_vec)):
 if kernels_vec[kernels_ind] == 'Gaussian':
 k_func = gaussian
 elif kernels_vec[kernels_ind] == 'Laplacian':
 k_func = triangle
 elif kernels_vec[kernels_ind] == 'Uniform':
 k_func = uniform
 elif kernels_vec[kernels_ind] == 'Triangular':
 k_func = laplacian
 else:
 k_func = epanechnikov

 for sigmas_ind in range(len(sigmas_vec)):

 for i, (train, test) in enumerate(skfold.split(data, target), start=0):
 x_train, x_test = data[train], data[test]
 y_train, y_test = target[train], target[test]
 result = pnn(x_train,y_train,x_test,k_func,sigmas_vec[sigmas_ind])
 n_test_samples = test.size
 PK_vec[i] = np.sum(result == y_test) / n_test_samples
 # print("Kernel = {}, Test #{:<2}: PK_vec {} test_size {}".format(b, i, PK_vec[i], n_test_samples))
 PK_krn_sig_vec[kernels_ind, sigmas_ind] = np.mean(PK_vec)
 print("Kernel = {}, sigma = {:2}:, PK {:<2}:".format(kernels_vec[kernels_ind], sigmas_vec[sigmas_ind],
PK_krn_sig_vec[kernels_ind, sigmas_ind]))

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(np.array(range(len(kernels_vec))), sigmas_vec)
surf = ax.plot_surface(X, Y, PK_krn_sig_vec.T, cmap='viridis')
ax.set_xlabel('kernel')
ax.set_xticks(np.array(range(len(kernels_vec))))
ax.set_xticklabels(kernels_vec)
ax.set_ylabel('sigma')
ax.set_zlabel('PK')
ax.view_init(30, 200)
# plt.savefig("Fig.1_PNN_new_CV_experiment.png",bbox_inches='tight')