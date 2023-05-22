import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    controls = np.loadtxt("control_out.csv", delimiter=",", dtype=float)
    trajectory = np.loadtxt("trajectory_out.csv", delimiter=",", dtype=float)
    time = np.zeros((1000,))
    for i in range(np.shape(controls)[0]):
        time[i] = 0.005 * i

    # Print Controls Over Time
    time = np.reshape(time, (1000, 1))
    controls = np.reshape(controls, (1000, 1))
    plt.plot(time, controls)
    plt.title('Controls Over Time')
    plt.xlabel("time - dt = 0.005 (s)")
    plt.ylabel("Controls (N)")
    plt.show()

    # Print Theta and Theta Dot Trajectory Over Time
    plt.plot(time, trajectory[:, 2])
    plt.title('Theta over Time')
    plt.xlabel("time - dt = 0.005 (s)")
    plt.ylabel("Theta (Rad)")
    plt.show()

    plt.plot(time, trajectory[:, 3])
    plt.title('Theta_dot over Time')
    plt.xlabel("time - dt = 0.005 (s)")
    plt.ylabel("Theta_dot (dRad/dt)")
    plt.show()
