#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import json
import os


class Plot2DMetric:
    def __init__(self, D, E, K, L, dt, dim1, dim2,
                 interpolation='none', vminmax=(-0.1, 0.5), sizexy=(15, 30)):
        """
        Plot Ergodic Spatial Metrix Phix using Demonstrations.

        - Can also plot the Spatial Information Density of Trajectory

        :param D: List of demonstrations (list)
        :param E: Weights that describe whether a demonstration D[i] is good [1] or bad [-1] (list)
        :param K: Size of series coefficient (int)
        :param L: Size of boundaries for dimensions [Lower boundary, Higher Boundary] (list)
        :param dt: Time difference (float)
        :param dim1: Dimension 1 to visualize State space - column # (int)
        :param dim2: Dimension 2 to visualize State space - column # (int)
        :param interpolation: Interpolation for Ergodic Metric Visualization (str)
        :param vminmax: Min/Max Value for matplotlib interpolation (vmin, vmax) (tuple)
        :param sizexy: Size of bins for x and y (dim1 bins, dim2 bins) (tuple)
        """
        self.K = K
        self.L = L

        self.dim1, self.dim2 = self.L[dim1], self.L[dim2]

        self.E = E
        D = np.array(D, dtype=object)
        self.D = [np.array(d)[:, [dim1, dim2]] for d in D]
        self.ergodic_measure = ErgodicMeasure(self.D, E, K, L, dt)
        self.phik_dict = self.ergodic_measure.get_phik(calc=True)

        self.Z = np.array([])

        self.interpolation = interpolation
        self.vminmax = vminmax
        self.sizexy = sizexy

    def visualize_ergodic(self):
        """
        Visualizes ergodic spatial metric Phix using inverse fourier transform of Phik.

        :return: None
        """
        Z = self.__calc_phix()
        plt.imshow(Z, interpolation=self.interpolation, vmin=self.vminmax[0], vmax=self.vminmax[1],
                   extent=[self.dim1[0], self.dim1[1], self.dim2[0], self.dim2[1]], aspect='auto')
        plt.title('Ergodic Metric Spatial Distribution')
        plt.xlabel("dim 1")
        plt.ylabel("dim 2")
        plt.show()

    def visualize_trajectory(self, show_trajectory=True, show_information_density=True):
        """
        Create a plot of all trajectory demonstrations.

        - Good demonstrations are blue o, bad demonstrations are red +
        - Shows information content of trajectory (positive values = positive demonstrations)

        :param show_trajectory:
        :param show_information_density:
        :return: None
        """
        if show_trajectory:
            for i in range(len(self.D)):
                if self.E[i] == 1:
                    plt.plot(self.D[i][:, 0], self.D[i][:, 1], 'bo',
                             markersize=0.2, linestyle='None')
                else:
                    plt.plot(self.D[i][:, 0], self.D[i][:, 1], 'r+',
                             markersize=0.2, linestyle='None')

        if show_information_density:
            contour_count = self.__calculate_contour_count()

            bin_theta = np.linspace(self.dim1[0], self.dim1[1], self.sizexy[0])
            bin_thetadot = np.linspace(self.dim2[0], self.dim2[1], self.sizexy[1])

            plt.contourf(bin_theta, bin_thetadot, contour_count, 100, cmap='RdBu', vmin=-6, vmax=6)
        plt.title('Spatial information density and trajectories')
        plt.xlabel("dim 1")
        plt.ylabel("dim 2")
        plt.show()

    def __calc_phix_x(self, x):
        """
        Calculate Phix variable using inverse fourier transform for position state vector x.

        Phix is defined by the function: Phix = 2 * Sum (phi_k * Fkx) for all k

        :param x: Position state vector x (np array of shape (1, 2))
        :return: Phi_x (float)
        """
        phix = 0
        for ki in range(self.K):
            for kj in range(self.K):
                k_str = f"{ki}{kj}"
                phik = self.phik_dict[k_str]
                fk = self.ergodic_measure.calc_Fk(x, [ki, kj])
                phix += phik * fk
        phix *= 2
        return phix

    def __calc_phix(self, axis_res=50):
        """
        Calculate Phix for all values of position vector x within bounds L1 and L2.

        :param axis_res: Number of discrete values for plot axis x and y (int)
        :return: Ergodic Measure Phix for all coordinates of dimension 1 and 2 in bounds L1 and L2
        """
        x_axis = np.linspace(self.dim1[0], self.dim1[1], axis_res)
        y_axis = np.linspace(self.dim2[0], self.dim2[1], axis_res)
        z = np.array([self.__calc_phix_x([i, j]) for j in y_axis for i in x_axis])
        Z = z.reshape(axis_res, axis_res)
        return Z

    def __calculate_contour_count(self):
        """
        Calculate information of all trajectory demonstrations using digitization within buckets.

        - Uses dimension 1 and 2 as bounds for bins, with self.sizexy[0] and self.sizexy[1]

        :return: contour_count - 2d information content in digitized buckets (np array)
        """
        bin_theta = np.linspace(self.dim1[0], self.dim1[1], self.sizexy[0])
        bin_thetadot = np.linspace(self.dim2[0], self.dim2[1], self.sizexy[1])
        contour_count = np.zeros((self.sizexy[1] + 1, self.sizexy[0] + 1))

        for i in range(len(self.D)):
            digitize_theta = np.digitize(self.D[i][:, 0], bin_theta)
            digitize_thetadot = np.digitize(self.D[i][:, 1], bin_thetadot)
            trajec_len = len(self.D[i])
            for ii in digitize_theta:
                for jj in digitize_thetadot:
                    contour_count[jj][ii] += (1 / trajec_len) * self.E[i]

        for i, row in enumerate(contour_count):
            for j, val in enumerate(row):
                if val > 10:
                    contour_count[i, j] = np.log10(val)
                elif val < -10:
                    contour_count[i, j] = -1 * np.log10(abs(val))
        contour_count = contour_count[:-1, :-1]
        return contour_count


class ErgodicMeasure:
    def __init__(self, D, E, K, L, dt):
        """
        Ergodic Helper Class to calculate the following metrics.

        - Phi_k: Spatial Distribution of demonstrations
        - h_k: Normalize factor for Fk
        - Lambda_k: Coefficient of Hilbert Space

        :param D: List of demonstrations (list)
        :param E: Weights that describe whether a demonstration D[i] is good [1] or bad [-1] (list)
        :param K: Size of series coefficient (int)
        :param L: Size of boundaries for dimensions, [Lower boundary, Higher Boundary] (list)
        :param dt: Time difference (float)
        """
        # Creating Fourier Distributions
        self.D = D
        self.E = E

        # Ergodic Measure variables
        self.K = K
        self.n = len(D[0][0])
        self.L = L  # needs upper and lower bound (L0, L1)
        self.dt = dt

        # Weights for demonstration trajectories
        self.m = len(self.D)
        self.w = np.array([(1/self.m) for _ in range(self.m)])

        # Stores lambda_k, phi_k, and ck values
        self.lambdak_values = {}
        self.phik_values = {}
        self.hk_values = {}

    def calc_fourier_metrics(self):
        """
        Calculate Phi_k, lambda_k, and h_k values based on demonstrations.

        :return: (hk_value (dict), phi_k value (dict), h_k value (dict))
        """
        self.__recursive_wrapper(self.K + 1, [], self.n, self.calc_phik)
        self.__recursive_wrapper(self.K + 1, [], self.n, self.calc_lambda_k)
        return self.hk_values, self.phik_values, self.lambdak_values

    def calc_phik(self, k):
        """
        Calculate coefficients that describe the task definition, phi_k.

        - Spatial distribution of demonstrations

        phi_k is defined by the following:
        phi_k = sum w_j * c_k_j where j ranges from 1 to num_trajectories
        - w_j is initialized as 1/num_trajectories, the weighting of each trajectory

        - Sets self.phi_k value in dictionary using k as a string
        :param k: k: The series coefficient given as a list of length dimensions (list)
        :return: None
        """
        phik = 0
        for i in range(self.m):
            phik += self.E[i] * self.w[i] * self.__calc_ck(self.D[i], k)
        k_str = ''.join(str(i) for i in k)
        self.phik_values[k_str] = phik

    def calc_lambda_k(self, k):
        """
        Calculate lambda_k places larger weights on lower coefficients of information.

        lambda_k is defined by the following:
        lambda_k = (1 + ||k||2) − s where s = n+1/2

        - Sets self.lambda_k value in dictionary using k as a string
        :param k: The series coefficient given as a list of length dimensions (list)
        :return: None
        """
        s = (self.n + 1) / 2
        lamnbda_k = 1 / ((1 + np.linalg.norm(k) ** 2) ** s)
        k_str = ''.join(str(i) for i in k)
        self.lambdak_values[k_str] = lamnbda_k

    def __calc_ck(self, x_t, k):
        """
        Calculate spacial statistics for a given trajectory and series coefficient value.

        ck is given by:
        ck = integral Fk(x(t)) dt from t=0 to t=T
        - where x(t) is a trajectory, mapping t to position vector x

        :param x_t: x(t) function, mapping position vectors over a period of time (np array)
        :param k: k: The series coefficient given as a list of length dimensions (list)
        :return: ck value (float)
        """
        x_len = len(x_t)
        T = x_len * self.dt
        Fk_x = np.zeros(x_len)
        for i in range(x_len):
            Fk_x[i] = self.calc_Fk(x_t[i], k)
        ck = (1 / T) * np.trapz(Fk_x, dx=self.dt)
        return ck

    def calc_Fk(self, x, k):
        """
        Calculate normalized fourier coeffecient using basis function metric.

        Fk is defined by the following:
        Fk = 1/hk * product(cos(k[i] *x[i])) where i ranges for all dimensions of x
        - Where k[i] = (K[i] * pi) / L[i]
        - Where L[i] is the bounds of the variable dimension i

        :param x: Position vector x (np array)
        :param k: The series coefficient given as a list of length dimensions (list)
        :return: Fk Value (float)
        """
        hk = self.__calc_hk(k)
        fourier_basis = 1
        for i in range(len(x)):
            fourier_basis *= np.cos((k[i]*np.pi*x[i])/(self.L[i][1] - self.L[i][0]))
        Fk = (1/hk)*fourier_basis
        return Fk

    def __calc_hk(self, k):
        """
        Normalize factor for Fk.

        hk is defined as:
        hk = Integral cos^2(k[i] * x[i]) dx from L[i][0] to L[i][1]

        :param k: The series coefficient given as a list of length dimensions (list)
        :return: hk value (float)
        """
        hk = 1
        for i in range(self.n):
            l0, l1 = self.L[i][0], self.L[i][1]
            if not k[i]:  # if k[i] is 0, we continue to avoid a divide by 0 error
                hk *= (l1 - l0)
                continue
            k_i = (k[i] * np.pi) / l1
            hk *= (2 * k_i * (l1 - l0) - np.sin(2 * k_i * l0) + np.sin(2 * k_i * l1)) / (4 * k_i)
        k_str = ''.join(str(i) for i in k)
        hk = np.sqrt(hk)
        self.hk_values[k_str] = hk
        return hk

    def __recursive_wrapper(self, K, k_arr, count, f):
        """
        Recurrsive wrapper allowing for to calculate various permuations of K.

        :param K: K Value - Needs to be passed as K+1 (int)
        :param k_arr: array of traversed K values (list)
        :param n: count of dimensions left to iterate through (int)
        :param f: function f to call with k_arr (function)
        :return:
        """
        if count > 0:
            for k in range(K):
                self.__recursive_wrapper(K, k_arr + [k], count - 1, f)
        else:
            print(k_arr)
            f(k_arr)

    def get_phik(self, calc=False):
        if calc:
            self.__recursive_wrapper(self.K + 1, [], self.n, self.calc_phik)
        return self.phik_values

    def get_hk(self):
        return self.hk_values

    def get_lambdak_values(self, calc=False):
        if calc:
            self.__recursive_wrapper(self.K + 1, [], self.n, self.calc_lambda_k)
        return self.lambdak_values


def main():
    print("Getting Demonstrations")
    ergodic_properties_json = input("Please input the full path to the ergodic properties json: ")
    # ergodic_properties_json = "../../cartpole/config/ergodic_properties.json"

    with open(ergodic_properties_json, 'r') as f:
        properties = json.load(f)
        ergodic_properties = properties["ergodic_system"]

    file_path = input("Please input a path to the demonstration folder: ")
    # file_path = "../../cartpole/demonstrations"
    # K = 15
    # L = [[-15, 15], [-15, 15], [-3.14159, 3.14159], [-11, 11]]
    # dt = 0.1
    # E = [1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1.2]
    K, L = ergodic_properties["K"], ergodic_properties["L"]
    dt, E = ergodic_properties["dt"], ergodic_properties["E"]
    demonstration_list, D, new_E = [], [], []

    input_demonstration = input("Please input the demonstrations you would like to use for \
                                training [list] "
                                "(if empty, all demonstrations are used) \ninput: ")
    if input_demonstration != "q" or len(input_demonstration) != 0:
        input_demonstration = input_demonstration.split(",")
        for num in input_demonstration:
            if num.isnumeric():
                demonstration_list.append(int(num))

    print(demonstration_list)
    sorted_files = sorted(os.listdir(file_path), key=lambda x: int(x[4:-4]))
    for i, file in enumerate(sorted_files):
        if (len(demonstration_list) != 0 and i not in demonstration_list) or "csv" not in file:
            continue
        new_E.append(E[i])
        demonstration_path = os.path.join(file_path, file)
        demonstration = np.genfromtxt(demonstration_path, delimiter=',')
        demonstration = np.hstack((demonstration[:, 2:], demonstration[:, :2]))
        D.append(demonstration)
    if len(D) == 0:
        raise FileNotFoundError("No files found in demonstration folder")

    print("Visualize Ergodic Metric")
    print(len(D))
    print(D[0][0])
    print(L[2])
    print(L[3])

    plot_phix_metric = Plot2DMetric(D, new_E, K, L, dt, 2, 3, interpolation='bilinear')
    plot_phix_metric.visualize_trajectory()
    plot_phix_metric.visualize_ergodic()



if __name__ == "__main__":
    main()
