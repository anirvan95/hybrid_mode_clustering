import matplotlib.pyplot as plt
import numpy as np
import scipy
import control.matlab as mt
from pydmd import DMDc
import pickle

from sklearn.mixture import BayesianGaussianMixture
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


def create_random_sys(mode, dim):
    sys_v = []
    for i in range(0, mode):
        sys_v.append(mt.drss(dim[0], dim[0], dim[1]))
    return sys_v


def create_traj(sys_v, dim, T):
    nmode = len(sys_v)
    x0 = np.array(np.random.rand(dim[0], 1))
    trajectory = [x0]
    input_v = []
    ltraj = 0
    for i in range(0, nmode):
        sys_id = np.random.randint(0, nmode)
        print(sys_id)
        segtime = int((T + np.random.randint(-5, 5)) / nmode)
        u = 0.5 * np.array(np.random.rand(dim[1], segtime))
        input_v.append(u)
        for j in range(0, segtime):  # variable time given to each dynamical system
            trajectory.append(sys_v[sys_id].A.dot(trajectory[:][ltraj]) + sys_v[sys_id].B.dot(u[:, j]).transpose())
            ltraj = ltraj + 1
        trajectory[:][ltraj] = trajectory[:][ltraj]

    trajectory = np.array(trajectory).T
    return {'trajectory': trajectory, 'input': input_v}


def smoothing(indices):
    newIndices = indices
    for i in range(1, len(indices) - 1):
        if indices[i] != indices[i - 1] and indices[i] != indices[i + 1] and indices[i + 1] == indices[i - 1]:
            newIndices[i] = indices[i + 1]

    return newIndices


def fitPolyRegression(traj, action, polydegree, transitions):
    nseg = len(transitions)
    dim = traj.shape[1]
    polynomial_features = PolynomialFeatures(degree=polydegree, interaction_only=True, include_bias=False)
    model = LinearRegression()
    dynamicMat = []
    rmse = 0
    selectedSeg = []
    for k in range(0, nseg - 1):
        if transitions[k + 1] - transitions[k] > 2:  # ensuring at least one sample is there between two transition point
            coeffVect = []
            print("Segment Number: ", k)
            selectedSeg.append(np.array([transitions[k], transitions[k + 1]]))

            for d in range(0, dim):
                y_target = traj[(transitions[k] + 1):transitions[k + 1], d]
                x_train = np.expand_dims(traj[transitions[k]:(transitions[k + 1] - 1), 1], axis=1)
                u_train = action[transitions[k]:(transitions[k + 1] - 1), :]
                feature_data_array = np.append(x_train, u_train, axis=1)
                feature_poly = polynomial_features.fit_transform(feature_data_array)
                model.fit(feature_poly, y_target)
                y_pred = model.predict(feature_poly)
                rmse = np.sqrt(mean_squared_error(y_target, y_pred))
                # plt.plot(y_target, 'r')
                # plt.plot(y_pred, 'b')
                # plt.show()
                print("RMSE : ", rmse)
                print("R2 score : ", r2_score(y_target,y_pred))
                # print(model.coef_)
                # print(model.intercept_)
                if d == 0:
                    coeffVect = np.array(model.coef_, model.intercept_)
                else:
                    coeffVect = np.hstack((coeffVect, model.coef_, model.intercept_))

            dynamicMat.append(coeffVect)

    return np.array(dynamicMat), np.array(selectedSeg)


def identifyTransitions(traj, window_size):
    total_size = traj.shape[0]
    dim = traj.shape[1]
    demo_data_array = np.zeros((total_size - window_size, dim * window_size))
    inc = 0
    for i in range(window_size, total_size):
        window = traj[i - window_size:i, :]
        demo_data_array[inc, :] = np.reshape(window, (1, dim * window_size))
        inc = inc + 1

    estimator = BayesianGaussianMixture(n_components=10, n_init=10, max_iter=300, weight_concentration_prior=1e-2,
                                        init_params='random', verbose=False)
    labels = estimator.fit_predict(demo_data_array)
    # print(estimator.weights_)
    filtabels = smoothing(labels)
    # print(labels)
    inc = 0
    transitions = []
    for j in range(window_size, total_size):

        if inc == 0 or j == window_size:
            pass  # self._transitions.append((i,0))
        elif j == (total_size - 1):
            pass  # self._transitions.append((i,n-1))
        elif filtabels[inc - 1] != filtabels[inc]:
            transitions.append(j - window_size)
        inc = inc + 1

    transitions.append(0)
    transitions.append(total_size - 1)
    transitions.sort()

    # print("[TSC] Discovered Transitions (time): ", transitions)
    return transitions


def getSeg(l, trajMat):
    count = 0
    for i in range(0, trajMat.shape[0]):
        for j in range(0, trajMat[i][1].shape[0]):
            if count == l:
                return i, j
            count = count + 1

    print("Error: Did not get segment !! ")
    return None


f = open("blocks_exp_raw_data_rs_1_mm_d40.p", "rb")
# data = pickle._Unpickler(f)
# data.encoding = 'latin1'
p = pickle.load(f, encoding='latin1')
f.close()

#Set parameters here
window_size = 3
rollout_data_array = []
ncomponents = 10
degree = 2
ndata = 20
rows = 3

trajMat = []
for rollout in range(0, ndata):
    print("Rollout Number", rollout, '\n')
    traj = p['X'][rollout, :, :]
    action = p['U'][rollout, :, :]

    tp = identifyTransitions(traj, window_size)

    X1 = [t[0] for t in traj]
    Y1 = [t[1] for t in traj]
    # plt.subplot(1, ndata, rollout + 1)
    plt.subplot(rows, ncomponents, rollout+1)
    plt.plot(X1, Y1, 'ro-')

    for i in range(0, len(tp)):
        point = traj[tp[i]]
        plt.subplot(rows, ncomponents, rollout+1)
        plt.plot(point[0], point[1], 'bo-')

    fittedModel, selTraj = fitPolyRegression(traj, action, degree, tp)
    trajMat.append(np.array([rollout, selTraj]))
    if rollout == 0:
        dynamicMat = fittedModel
    else:
        dynamicMat = np.concatenate((dynamicMat, fittedModel), axis=0)

trajMat = np.array(trajMat)
print(trajMat.shape)
print(np.array(dynamicMat).shape)

estimator = BayesianGaussianMixture(n_components=ncomponents,  n_init=10, max_iter=300, weight_concentration_prior=0.5, init_params='random', verbose=False)
labels = estimator.fit_predict(np.array(dynamicMat))
print(labels)
weight_vector = np.array(estimator.weights_)
print(estimator.weights_)

for ncomp in range(0, ncomponents):
    print("Showing segments for ", ncomp, " cluster ")
    for l in range(0, len(labels)):
        if ncomp == labels[l]:
            rt, segtra = getSeg(l, trajMat)
            exTraj = p['X'][rt, :, :]
            X1 = exTraj[trajMat[rt][1][segtra][0]:trajMat[rt][1][segtra][1], 0]
            Y1 = exTraj[trajMat[rt][1][segtra][0]:trajMat[rt][1][segtra][1], 1]
            plt.subplot(rows, ncomponents, labels[l]+rollout+3)
            plt.plot(X1, Y1, 'r')

plt.show()
