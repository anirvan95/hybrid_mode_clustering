import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import DBSCAN
from math import sqrt
from sklearn.preprocessing import normalize
import sys

# supressing warnings
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


def computeDistance(f1, f2):
    dim = int(0.5 * (-1 + sqrt(1 + 4 * len(f1))))
    mf1 = f1[0:dim]
    mf2 = f2[0:dim]
    covf1 = np.reshape(f1[dim:], (-1, dim))
    covf2 = np.reshape(f2[dim:], (-1, dim))
    return .5 * (bhattacharyyaGaussian(mf1, covf1, mf2, covf2) + bhattacharyyaGaussian(mf2, covf2, mf1, covf1))


def bhattacharyyaGaussian(pm, pv, qm, qv):
    """
    Computes Bhattacharyya distance between two Gaussians
    with diagonal covariance.
    """
    # Difference between means pm, qm
    diff = np.expand_dims((qm - pm), axis=1)
    # Interpolated variances
    pqv = (pv + qv) / 2.
    # Log-determinants of pv, qv
    ldpv = np.linalg.det(pv)
    ldqv = np.linalg.det(qv)
    # Log-determinant of pqv
    ldpqv = np.linalg.det(pqv)
    # "Shape" component (based on covariances only)
    # 0.5 log(|\Sigma_{pq}| / sqrt(\Sigma_p * \Sigma_q)
    norm = 0.5 * np.log(ldpqv/(np.sqrt(ldpv*ldqv)))
    # "Divergence" component (actually just scaled Mahalanobis distance)
    # 0.125 (\mu_q - \mu_p)^T \Sigma_{pq}^{-1} (\mu_q - \mu_p)
    temp = np.matmul(diff.transpose(), np.linalg.pinv(pqv))
    dist = 0.125 * np.matmul(temp, diff)
    return np.float(dist + norm)


def fitGaussianDistribution(traj, action, transitions):
    """
        Fits gaussian distribution in each segment of the trajectory
    """
    nseg = len(transitions)
    dim = traj.shape[1]
    dynamicMat = []
    rmse = 0
    selectedSeg = []
    for k in range(0, nseg - 1):
        if transitions[k + 1] - transitions[k] > 2:
            # ensuring at least one sample is there between two transition point
            x_t_1 = traj[(transitions[k] + 1):transitions[k + 1], :]
            x_t = traj[transitions[k]:(transitions[k + 1] - 1), :]
            u_t = action[transitions[k]:(transitions[k + 1] - 1), :]
            feature_data_array = np.hstack((x_t, x_t_1))
            meanGaussian = np.mean(feature_data_array, axis=0)
            covGaussian = np.cov(feature_data_array, rowvar=0)
            covFeature = covGaussian.flatten()
            det = np.linalg.det(covGaussian)
            if np.linalg.cond(covGaussian) < 1/sys.float_info.epsilon:
                # print("Segment Number: ", k)
                selectedSeg.append(np.array([transitions[k], transitions[k + 1]]))
                dynamicMat.append(np.append(meanGaussian, covGaussian))
            else:
                print("Singular Matrix !!! ")

    return np.array(dynamicMat), np.array(selectedSeg)


def smoothing(indices):
    """
        Smoothing for transition point detection [IMPROVE]
    """
    newIndices = indices
    for i in range(1, len(indices) - 1):
        if indices[i] != indices[i - 1] and indices[i] != indices[i + 1] and indices[i + 1] == indices[i - 1]:
            newIndices[i] = indices[i + 1]

    return newIndices


def identifyTransitions(traj, window_size):
    """
        Identify transition by accumulating data points using sliding window and using DP GMM to find
        clusters in a single trajectory
    """
    total_size = traj.shape[0]
    dim = traj.shape[1]
    demo_data_array = np.zeros((total_size - window_size, dim * window_size))
    inc = 0
    for i in range(window_size, total_size):
        window = traj[i - window_size:i, :]
        demo_data_array[inc, :] = np.reshape(window, (1, dim * window_size))
        inc = inc + 1

    estimator = BayesianGaussianMixture(n_components=5, n_init=10, max_iter=300, weight_concentration_prior=0.001,
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

    print("[TSC] Discovered Transitions (number): ", len(transitions))
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


f = open("yumi_peg_exp_new_raw_data_train.p", "rb")
p = pickle.load(f, encoding='latin1')
f.close()

window_size = 2
rollout_data_array = []
ncomponents = 10
degree = 2
ndata = 15
rows = 3
cols = 10

dt = p['exp_params']['dt']

# viscous friction
Fv = np.array([1.06, 1.09, 0.61, 0.08, 0.08, 0.08, 0.52])
# static friction
Fc = np.array([2.43, 2.76, 1.11, 0.52, 0.52, 0.52, 1.00])

trajMat = []

for rollout in range(0, ndata):
    joint_torques = p['U'][rollout, :, :]
    joint_angles = p['X'][rollout, :, 0:7]
    joint_velocity = p['X'][rollout, :, 7:14]
    cart_force = p['F'][rollout, :, :]
    cart_position = p['EX'][rollout, :, :]
    kinetic_energy_vect = []
    input_energy_vect = []
    frictional_torque_vect = []
    feature_vect = []
    input_energy = 0

    # calculating energy for each rollout

    for i in range(0, len(joint_torques)):
        frictional_torque = np.multiply(Fv, joint_velocity[i, :])
        input_energy = input_energy + np.matmul((joint_torques[i, :] - frictional_torque), joint_velocity[i, :].T) * dt
        feature = np.hstack((input_energy, cart_position[i, 0]))
        feature_vect.append(feature)
        input_energy_vect.append(input_energy)
        frictional_torque_vect.append(frictional_torque)

    # traj = np.expand_dims(np.array(feature_vect), axis=1)
    tp_traj = np.array(feature_vect)
    tp = identifyTransitions(tp_traj, window_size)
    # plot transitions

    plt.subplot(rows, cols, rollout + 1)
    plt.plot(tp_traj[:, 0], 'r')

    for i in range(0, len(tp)):
        point = tp_traj[tp[i], :]
        plt.subplot(rows, cols, rollout + 1)
        plt.plot(tp[i], point[0], 'bo-')

    # create trajectory of joint velocity, joint angles and cartesian positions
    traj = cart_position[:, 0:4]
    action = joint_torques
    fittedModel, selTraj = fitGaussianDistribution(traj, action, tp)
    # storing the fitted distribution for all the rollouts
    trajMat.append(np.array([rollout, selTraj]))
    if rollout == 0:
        dynamicMat = fittedModel
    else:
        dynamicMat = np.concatenate((dynamicMat, fittedModel), axis=0)

print(np.array(dynamicMat).shape)
print(np.array(trajMat).shape)

trajMat = np.array(trajMat)
# DBSCAN based clustering

db = DBSCAN(eps=5, min_samples=2, metric=computeDistance)
labels = db.fit_predict(dynamicMat)
print(labels)
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

ncomponents = len(np.unique(labels))
for ncomp in range(0, ncomponents):
    print("Showing segments for ", ncomp, " cluster ")
    for l in range(0, len(labels)):
        if ncomp == labels[l]:
            rt, segtra = getSeg(l, trajMat)
            exTraj = p['EX'][rt, :, :]
            X1 = exTraj[trajMat[rt][1][segtra][0]:trajMat[rt][1][segtra][1], :]
            plt.subplot(rows, cols, labels[l] + rollout + 3)
            plt.plot(X1, 'r')

plt.show()
