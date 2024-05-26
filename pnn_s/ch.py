import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import hickle as hkl

# Wczytanie danych z pliku CSV
file_path = 'sonar.all-data'
data = pd.read_csv(file_path, header=None)

# Zakładam, że ostatnia kolumna to etykiety
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Konwersja etykiet tekstowych na numeryczne
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Parametry modelu PNN
CVN = 50
skfold = StratifiedKFold(n_splits=CVN)
PK_vec = np.zeros(CVN)
uniform = lambda x, b: (np.abs(x / b) <= 1) and 1 / 2 or 0
triangle = lambda x, b: (np.abs(x / b) <= 1) and (1 - np.abs(x / b)) or 0
gaussian = lambda x, b: (1.0 / np.sqrt(2 * np.pi)) * np.exp(-.5 * (x / b) ** 2)
laplacian = lambda x, b: (1.0 / (2 * b)) * np.exp(-np.abs(x / b))
epanechnikov = lambda x, b: (np.abs(x / b) <= 1) and ((3 / 4) * (1 - (x / b) ** 2)) or 0

def pattern_layer(inp, kernel, sigma, x_train):
    k_values = []
    for p in x_train:
        edis = np.linalg.norm(p - inp)  # find euclidean distance
        k = kernel(edis, sigma)  # pass values of euclidean dist and smoothing parameter to kernel function
        k_values.append(k)
    return k_values

def summation_layer(k_values, Y_train, class_uniques, class_counts):
    summed = np.zeros(len(class_uniques))
    for i, c in enumerate(class_uniques):
        val = (Y_train == class_uniques[i])
        k_values = np.array(k_values)
        summed[i] = np.sum(k_values[val])
    avg_sum = list(summed / class_counts)
    return avg_sum

def output_layer(avg_sum, class_uniques):
    maxv = max(avg_sum)
    label = class_uniques[avg_sum.index(maxv)]
    return label

def pnn(X_train, Y_train, X_test, kernel, sigma):
    class_uniques = np.unique(Y_train)
    class_counts = np.zeros(len(class_uniques))
    for i, class_val in enumerate(class_uniques):
        class_counts[i] = sum(class_val == Y_train)
    labels = []
    for s in X_test:
        k_values = pattern_layer(s, kernel, sigma, X_train)
        avg_sum = summation_layer(k_values, Y_train, class_uniques, class_counts)
        label = output_layer(avg_sum, class_uniques)
        labels.append(label)
    return labels

# Candidate Kernels
kernels_vec = ["Gaussian", "Triangular", "Epanechnikov", "Uniform", "Laplacian"]
sigmas_vec = np.arange(0.1, 2.2, 0.2)
PK_krn_sig_vec = np.zeros([len(kernels_vec), len(sigmas_vec)])

for kernels_ind in range(len(kernels_vec)):
    if kernels_vec[kernels_ind] == 'Gaussian':
        k_func = gaussian
    elif kernels_vec[kernels_ind] == 'Laplacian':
        k_func = laplacian
    elif kernels_vec[kernels_ind] == 'Uniform':
        k_func = uniform
    elif kernels_vec[kernels_ind] == 'Triangular':
        k_func = triangle
    else:
        k_func = epanechnikov

    for sigmas_ind in range(len(sigmas_vec)):
        for i, (train, test) in enumerate(skfold.split(X, y), start=0):
            x_train, x_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            result = pnn(x_train, y_train, x_test, k_func, sigmas_vec[sigmas_ind])
            n_test_samples = test.size
            PK_vec[i] = np.sum(result == y_test) / n_test_samples
        PK_krn_sig_vec[kernels_ind, sigmas_ind] = np.mean(PK_vec)
        print(f"Kernel = {kernels_vec[kernels_ind]}, sigma = {sigmas_vec[sigmas_ind]:.2f}, PK = {PK_krn_sig_vec[kernels_ind, sigmas_ind]:.4f}")

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
X_plot, Y_plot = np.meshgrid(np.array(range(len(kernels_vec))), sigmas_vec)
surf = ax.plot_surface(X_plot, Y_plot, PK_krn_sig_vec.T, cmap='viridis')
ax.set_xlabel('kernel')
ax.set_xticks(np.array(range(len(kernels_vec))))
ax.set_xticklabels(kernels_vec)
ax.set_ylabel('sigma')
ax.set_zlabel('PK')
ax.view_init(30, 200)
plt.show()
