import matplotlib.pyplot as plt
import numpy as np
import scipy
import control.matlab as mt
from pydmd import DMDc
import pickle

from sklearn.mixture import BayesianGaussianMixture
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
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


'''
def fitLinModel(traj, action, transitions):
    dmdc = DMDc(svd_rank=-1)
    model = np.zeros((len(transitions), 1))
    for k in range(0, len(transitions) - 1):
        dmdc.fit(traj[transitions[k]:transitions[k + 1], :], action[transitions[k]:transitions[k + 1], :])
        model[k] = dmdc.eigs
    return model
'''


def fitPolyRegression(traj, action, polydegree, transitions):
    nseg = len(transitions)
    dim = traj.shape[1]
    polynomial_features = PolynomialFeatures(degree=polydegree)
    model = LinearRegression()
    dynamicMat = []
    for k in range(0, nseg-1):
        if transitions[k + 1] - transitions[k] > 1:  # ensuring at least one sample is there between two transition point
            coeffVect = []
            for d in range(0, dim):
                y_target = traj[(transitions[k]+1):transitions[k + 1], d]
                x_train = traj[transitions[k]:(transitions[k + 1]-1), :]
                u_train = action[transitions[k]:(transitions[k + 1]-1), :]
                feature_data_array = np.append(x_train, u_train, axis=1)
                feature_poly = polynomial_features.fit_transform(feature_data_array)
                model.fit(feature_poly, y_target)
                if d == 0:
                    coeffVect = np.array(model.coef_)
                else:
                    coeffVect = np.hstack((coeffVect, model.coef_))

            dynamicMat.append(coeffVect)

    return np.array(dynamicMat)


def identifyTransitions(traj, window_size):
    total_size = traj.shape[0]
    dim = traj.shape[1]
    demo_data_array = np.zeros((total_size - window_size, dim * window_size))
    inc = 0
    for i in range(window_size, total_size):
        window = traj[i - window_size:i, :]
        demo_data_array[inc, :] = np.reshape(window, (1, dim * window_size))
        inc = inc + 1

    estimator = BayesianGaussianMixture(n_components=10, n_init=10, max_iter=300, degrees_of_freedom_prior=4, weight_concentration_prior=1e-1, init_params='random', verbose=False)
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


f = open("blocks_exp_raw_data_rs_1_mm_d40.p", "rb")
# data = pickle._Unpickler(f)
# data.encoding = 'latin1'
p = pickle.load(f, encoding='latin1')
f.close()
window_size = 2
rollout_data_array = []


traj = p['X'][5, :, :]
action = p['U'][5, :, :]
tp = identifyTransitions(traj, window_size)

ndata = 50


for rollout in range(0, ndata):
    print("Rollout Number", rollout, '\n')
    traj = p['X'][rollout, :, :]
    action = p['U'][rollout, :, :]

    tp = identifyTransitions(traj, window_size)
    fittedModel = fitPolyRegression(traj, action, 2, tp)
    if rollout == 0:
        dynamicMat = fittedModel
    else:
        dynamicMat = np.concatenate((dynamicMat, fittedModel), axis=0)

# print(rollout_data_array.shape)

print(np.array(dynamicMat).shape)
estimator = BayesianGaussianMixture(n_components=10, n_init=10, max_iter=300, weight_concentration_prior=1e-2, init_params='random', verbose=False)
labels = estimator.fit_predict(np.array(dynamicMat))
print(np.unique(labels))
print(estimator.weights_)
'''

 X1 = [t[0] for t in traj]
    Y1 = [t[1] for t in traj]
    plt.subplot(1, ndata, rollout+1)
    plt.plot(X1, Y1, 'ro-')

    for i in range(0, len(tp)):
        point = traj[tp[i]]
        plt.subplot(1, ndata, rollout+1)
        plt.plot(point[0], point[1], 'bo-')
        
        
# print(linmod.shape)
X1 = [t[0] for t in traj]
Y1 = [t[1] for t in traj]
plt.plot(X1, Y1, 'ro-')

for i in range(0, len(tp)):
    point = traj[tp[i]]
    plt.plot(point[0], point[1], 'bo-')
'''
plt.show()

