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
from sklearn.cluster import DBSCAN
from math import sqrt


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


def kl_mvn(m0, S0, m1, S1):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.

    - accepts stacks of means, but only one S0 and S1

    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| +
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term = np.trace(iS1 @ S0)
    det_term = np.log(np.linalg.det(S1) / np.linalg.det(S0))  # np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ np.linalg.inv(S1) @ diff  # np.sum( (diff*diff) * iS1, axis=1)
    return .5 * (tr_term + det_term + quad_term - N)


def gau_bh(pm, pv, qm, qv):
    """
    Classification-based Bhattacharyya distance between two Gaussians
    with diagonal covariance.  Also computes Bhattacharyya distance
    between a single Gaussian pm,pv and a set of Gaussians qm,qv.
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
    temp = np.matmul(diff.transpose(), np.linalg.inv(pqv))
    dist = 0.125 * np.matmul(temp, diff)
    return np.float(dist + norm)


def fitGaussianDistribution(traj, action, transitions):
    nseg = len(transitions)
    dim = traj.shape[1]
    dynamicMat = []
    rmse = 0
    selectedSeg = []
    for k in range(0, nseg - 1):
        if transitions[k + 1] - transitions[k] > 2:  # ensuring at least one sample is there between two transition point

            x_t_1 = traj[(transitions[k] + 1):transitions[k + 1], :]
            x_t = traj[transitions[k]:(transitions[k + 1] - 1), :]
            u_t = action[transitions[k]:(transitions[k + 1] - 1), :]
            feature_data_array = np.hstack((x_t, u_t, x_t_1))
            meanGaussian = np.mean(feature_data_array, axis=0)
            covGaussian = np.cov(feature_data_array, rowvar=0)
            covFeature = covGaussian.flatten()
            det = np.linalg.det(covGaussian)
            if det != 0:
                # print("Segment Number: ", k)
                selectedSeg.append(np.array([transitions[k], transitions[k + 1]]))
                dynamicMat.append(np.append(meanGaussian, covGaussian))
            else:
                print("Singular Matrix !!! ")

    return np.array(dynamicMat), np.array(selectedSeg)


# try to make it general, depends on the feature vector
def KLDdistance(f1, f2):
    dim = int(0.5 * (-1 + sqrt(1 + 4 * len(f1))))
    mf1 = f1[0:dim]
    mf2 = f2[0:dim]
    covf1 = np.reshape(f1[dim:], (-1, dim))
    covf2 = np.reshape(f2[dim:], (-1, dim))
    return .5 * (gau_bh(mf1, covf1, mf2, covf2) + gau_bh(mf2, covf2, mf1, covf1))


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
                print("R2 score : ", r2_score(y_target, y_pred))
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

# Set parameters here
window_size = 3
rollout_data_array = []
ncomponents = 10
degree = 2
ndata = 50
# for plotting
rows = 5
cols = 15

trajMat = []
for rollout in range(0, ndata):
    print("Rollout Number", rollout, '\n')
    traj = p['X'][rollout, :, :]
    action = p['U'][rollout, :, :]

    tp = identifyTransitions(traj, window_size)

    X1 = [t[0] for t in traj]
    Y1 = [t[1] for t in traj]
    # plt.subplot(1, ndata, rollout + 1)
    plt.subplot(rows, cols, rollout + 1)
    plt.plot(X1, Y1, 'ro-')

    for i in range(0, len(tp)):
        point = traj[tp[i]]
        plt.subplot(rows, cols, rollout + 1)
        plt.plot(point[0], point[1], 'bo-')

    # fittedModel, selTraj = fitPolyRegression(traj, action, degree, tp)
    fittedModel, selTraj = fitGaussianDistribution(traj, action, tp)
    trajMat.append(np.array([rollout, selTraj]))
    if rollout == 0:
        dynamicMat = fittedModel
    else:
        dynamicMat = np.concatenate((dynamicMat, fittedModel), axis=0)


trajMat = np.array(trajMat)
print(trajMat.shape)
print(np.array(dynamicMat).shape)

# DPGMM based clustering
'''
estimator = BayesianGaussianMixture(n_components=ncomponents, n_init=10, max_iter=300, weight_concentration_prior=1,
                                    init_params='random', verbose=False)
labels = estimator.fit_predict(np.array(dynamicMat))
print(labels)
weight_vector = np.array(estimator.weights_)
print(estimator.weights_)
'''

# DBSCAN based clustering
db = DBSCAN(eps=10, min_samples=2, metric=KLDdistance)
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
            exTraj = p['X'][rt, :, :]
            X1 = exTraj[trajMat[rt][1][segtra][0]:trajMat[rt][1][segtra][1], 0]
            Y1 = exTraj[trajMat[rt][1][segtra][0]:trajMat[rt][1][segtra][1], 1]
            plt.subplot(rows, cols, labels[l] + rollout + 3)
            plt.plot(X1, Y1, 'r')

plt.show()
