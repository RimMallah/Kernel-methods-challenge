import numpy as np
import pandas as pd
from time import time
import sys
from joblib import Parallel, delayed
from cvxopt import matrix as cvxMat, solvers as QPSolver


## SIFT
def array_data(X_test):
    red, green, blue = np.hsplit(X_test, 3)
    data_test = np.array([np.dstack((red[i], blue[i], green[i])).reshape(32, 32, 3) for i in range(len(X_test))])
    return data_test

def get_weights(num_bins, ps):
    size_unit = np.array(range(ps))
    sph, spw = np.meshgrid(size_unit, size_unit)
    sph.resize(sph.size)
    spw.resize(spw.size)
    bincenter = np.array(range(1, num_bins*2, 2)) / 2.0 / num_bins * ps - 0.5
    bincenter_h, bincenter_w = np.meshgrid(bincenter, bincenter)
    bincenter_h.resize((bincenter_h.size, 1))
    bincenter_w.resize((bincenter_w.size, 1))
    dist_ph = abs(sph - bincenter_h)
    dist_pw = abs(spw - bincenter_w)
    weights_h = dist_ph / (ps / np.double(num_bins))
    weights_w = dist_pw / (ps / np.double(num_bins))
    weights_h = (1-weights_h) * (weights_h <= 1)
    weights_w = (1-weights_w) * (weights_w <= 1)
    return weights_h * weights_w

def convolution2D(image, kernel):
    imRows, imCols = image.shape
    kRows, kCols = kernel.shape
    y = np.zeros((imRows,imCols))
    kcenterX = kCols//2
    kcenterY = kRows//2
    for i in range(imRows):
        for j in range(imCols):
            for m in range(kRows):
                mm = kRows - 1 - m
                for n in range(kCols):
                    nn = kCols - 1 - n
                    ii = i + (m - kcenterY)
                    jj = j + (n - kcenterX)
                    if ii >= 0 and ii < imRows and jj >= 0 and jj < imCols :
                        y[i][j] += image[ii][jj] * kernel[mm][nn]
    return y

def log_process(title, cursor, finish_cursor, start_time = None):
    percentage = float(cursor + 1)/finish_cursor
    now_time = time()
    time_to_finish = ((now_time - start_time)/percentage) - (now_time - start_time)
    mn, sc = int(time_to_finish//60), int((time_to_finish/60 - time_to_finish//60)*60)
    if start_time:
        sys.stdout.write("\r%s - %.2f%% ----- Temps restant estimé: %d min %d sec -----" %(title, 100*percentage, mn, sc))
        sys.stdout.flush()
    else:
        sys.stdout.write("\r%s - \r%.2f%%" %(title, 100*percentage))
        sys.stdout.flush()
        
        
class SIFT:
    def __init__(self, gs=8, ps=16, gaussian_thres=1.0, gaussian_sigma=0.8, sift_thres=0.2,
                 num_angles=12, num_bins=5, alpha=9.0):
        self.num_angles = num_angles
        self.num_bins = num_bins
        self.alpha = alpha
        self.angle_list = np.array(range(num_angles)) * 2.0 * np.pi / num_angles
        self.gs = gs
        self.ps = ps
        self.gaussian_thres = gaussian_thres
        self.gaussian_sigma = gaussian_sigma
        self.sift_thres = sift_thres
        self.weights = get_weights(num_bins, ps)

    def get_params_image(self, image):
        image = image.astype(np.double)
        if image.ndim == 3:
            image = np.mean(image, axis=2)
        H, W = image.shape
        gS = self.gs
        pS = self.ps
        remH = np.mod(H-pS, gS)
        remW = np.mod(W-pS, gS)
        offsetH = remH//2
        offsetW = remW//2
        gridH, gridW = np.meshgrid(range(offsetH, H-pS+1, gS), range(offsetW, W-pS+1, gS))
        gridH = gridH.flatten()
        gridW = gridW.flatten()
        features = self._calculate_sift_grid(image, gridH, gridW)
        features = self._normalize_sift(features)
        positions = np.vstack((gridH / np.double(H), gridW / np.double(W)))
        return features, positions
    
    def get_X(self, data):
        out = []
        start = time()
        finish = len(data)
        for idx, dt in enumerate(data):
            log_process('SIFT', idx, finish_cursor=finish, start_time = start)
            out.append(self.get_params_image(np.mean(np.double(dt), axis=2))[0][0])
        return np.array(out)

    def _calculate_sift_grid(self, image, gridH, gridW):
        H, W = image.shape
        Npatches = gridH.size
        features = np.zeros((Npatches, self.num_bins * self.num_bins * self.num_angles))
        gaussian_height, gaussian_width = self._get_gauss_filter(self.gaussian_sigma)
        IH = convolution2D(image, gaussian_height)
        IW = convolution2D(image, gaussian_width)
        Imag = np.sqrt(IH**2 + IW**2)
        Itheta = np.arctan2(IH, IW)
        Iorient = np.zeros((self.num_angles, H, W))
        for i in range(self.num_angles):
            Iorient[i] = Imag * np.maximum(np.cos(Itheta - self.angle_list[i])**self.alpha, 0)
        for i in range(Npatches):
            currFeature = np.zeros((self.num_angles, self.num_bins**2))
            for j in range(self.num_angles):
                currFeature[j] = np.dot(self.weights,\
                        Iorient[j,gridH[i]:gridH[i]+self.ps, gridW[i]:gridW[i]+self.ps].flatten())
            features[i] = currFeature.flatten()
        return features

    def _normalize_sift(self, features):
        siftlen = np.sqrt(np.sum(features**2, axis=1))
        hcontrast = (siftlen >= self.gaussian_thres)
        siftlen[siftlen < self.gaussian_thres] = self.gaussian_thres
        features /= siftlen.reshape((siftlen.size, 1))
        features[features>self.sift_thres] = self.sift_thres
        features[hcontrast] /= np.sqrt(np.sum(features[hcontrast]**2, axis=1)).\
                reshape((features[hcontrast].shape[0], 1))
        return features

    def _get_gauss_filter(self, sigma):
        gaussian_filter_amp = int(2*np.ceil(sigma))
        gaussian_filter = np.array(range(-gaussian_filter_amp, gaussian_filter_amp+1))**2
        gaussian_filter = gaussian_filter[:, np.newaxis] + gaussian_filter
        gaussian_filter = np.exp(- gaussian_filter / (2.0 * sigma**2))
        gaussian_filter /= np.sum(gaussian_filter)
        gaussian_height, gaussian_width = np.gradient(gaussian_filter)
        gaussian_height *= 2.0/np.sum(np.abs(gaussian_height))
        gaussian_width  *= 2.0/np.sum(np.abs(gaussian_width))
        return gaussian_height, gaussian_width
    
    
## Kernel

class RBF:
    def __init__(self, sigma=0.7):
        self.sigma = sigma
    def kernel(self, X, Y):
        G = np.sum(X**2,-1) + np.sum(Y**2,-1) - 2*np.dot(X,Y.T)
        return np.exp(-G/(2*self.sigma**2))
    

## SVC

# Binary Support Vector Classification
class KernelSVC_Binary:
    def __init__(self, kernel, C=1.0, threshold=1e-5):
        self.C = float(C)
        self.threshold = threshold
        self.kernel = kernel

    def fit(self, X, y, K):
        y = np.where(y == 0, -1, y)  # Convert binary labels from {0, 1} to {-1, 1}
        y = np.array(y, dtype=np.double)
        P = cvxMat(np.outer(y, y) * K)
        q = cvxMat(-np.ones(len(X)))
        G = cvxMat(np.vstack([-np.eye(len(X)), np.eye(len(X))]))
        h = cvxMat(np.hstack([np.zeros(len(X)), np.ones(len(X)) * self.C]))
        A = cvxMat(y.reshape(1, -1))  # Ensure y is shaped correctly
        b = cvxMat(0.0)
        solution = QPSolver.qp(P, q, G, h, A, b)
        a = np.ravel(solution['x'])
        support = a > self.threshold
        self.a = a[support]
        self.support = X[support]
        self.support_y = y[support]
        self.b = np.mean(self.support_y - np.sum(self.a * self.support_y * K[support][:, support], axis=1))

    def project(self, X):
        return np.array([sum(a * support_y * self.kernel(x, support) for a, support, support_y in zip(self.a, self.support, self.support_y)) for x in X]) + self.b

    def predict(self, X):
        return np.sign(self.project(X))

# One vs Rest for multi-class classification
class KernelSVC(KernelSVC_Binary):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nb_classes = 10

    def fit(self, X, y):
        self.K = self.kernel(X, X)
        
        # Define a separate function to fit each classifier and return it
        def fit_and_return(model, X, y, K):
            model.fit(X, y, K)
            return model

        # Parallel execution
        self.models_ = Parallel(n_jobs=-1, verbose=11)(
            delayed(fit_and_return)(
                KernelSVC_Binary(kernel=self.kernel, C=self.C),
                X, np.where(y == i, 1, 0), self.K
            ) for i in range(self.nb_classes)
        )
    
    def predict(self, X):
        # Define a separate function to make predictions for a classifier and return them
        def predict_and_return(model, X):
            return model.project(X)

        # Parallel execution
        predictions = Parallel(n_jobs=-1, verbose=11)(
            delayed(predict_and_return)(model, X) for model in self.models_
        )

        predictions = np.array(predictions).T
        return np.argmax(predictions, axis=1)


    
## Apply Pipeline

Xtr = np.array(pd.read_csv('data/Xtr.csv',header=None,sep=',',usecols=range(3072)))
Ytr = np.array(pd.read_csv('data/Ytr.csv',sep=',',usecols=[1]), dtype=int).squeeze() 

X = array_data(Xtr)
Y = Ytr

# # Apply data augmentation
# Xtr_rot = np.array([np.rot90(img, k=np.random.randint(1, 4)) for img in Xtr])
# Xtr_trans = np.array([np.roll(img, shift=np.random.randint(-1, 2), axis=0) for img in Xtr])

# X = np.vstack((Xtr, Xtr_rot, Xtr_trans))
# Y = np.tile(Ytr, 3)

# indices = np.arange(X.shape[0])
# np.random.shuffle(indices)
# X = X[indices]
# Y = Y[indices]

print(X.shape, Y.shape)

sift = SIFT(gs=6,
            ps=31,
            sift_thres=.3,
            gaussian_sigma=.4,
            gaussian_thres=.7,
            num_angles=12,
            num_bins=5,
            alpha=9.0)

X = sift.get_X(X)

kernel = RBF(sigma=.7).kernel
svc = KernelSVC(kernel=kernel, C=7.0)
svc.fit(X, Y)

Xte = np.array(pd.read_csv('data/Xte.csv',header=None,sep=',',usecols=range(3072)), dtype=float)

X_test = sift.get_X(array_data(Xte))

Y_pred = svc.predict(X_test)

Yte = {'Prediction': Y_pred}
dataframe = pd.DataFrame(Yte)
dataframe.index += 1
dataframe.to_csv('Yte.csv',index_label='Id')