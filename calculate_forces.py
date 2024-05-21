"""

@author: Artur Ruppel

"""

import numpy as np
from scipy.fft import fft2, ifft2

def calculate_traction_stresses(u, v, E, nu, pixelsize, alpha):
    '''This function takes a displacement field u and v, a gel rigidity E, it's poisson ratio nu, the size of a pixel in '''
    M, N = u.shape

    # pad displacement map with zeros until it has a shape of 2^n by 2^n because fourier transform is faster
    n = 2
    while (2 ** n < M) or (2 ** n < N):
        n = n + 1

    M2 = 2 ** n
    N2 = M2

    u_padded = np.zeros((M2, N2))
    v_padded = np.zeros((M2, N2))

    u_padded[:u.shape[0], :u.shape[1]] = u
    v_padded[:v.shape[0], :v.shape[1]] = v

    u_fft = fft2(u_padded)
    v_fft = fft2(v_padded)

    # remove component related to translation
    u_fft[0, 0] = 0
    v_fft[0, 0] = 0

    Kx1 = (2 * np.pi / pixelsize) / N2 * np.arange(int(N2 / 2))
    Kx2 = -(2 * np.pi / pixelsize) / N2 * (N2 - np.arange(int(N2 / 2), N2))
    Ky1 = (2 * np.pi / pixelsize) / M2 * np.arange(int(M2 / 2))
    Ky2 = -(2 * np.pi / pixelsize) / M2 * (M2 - np.arange(int(M2 / 2), M2))

    Kx = np.concatenate((Kx1, Kx2))
    Ky = np.concatenate((Ky1, Ky2))

    kx, ky = np.meshgrid(Kx, Ky)
    k = np.sqrt(kx ** 2 + ky ** 2)
    t_xt = np.zeros((M2, N2), dtype=complex)
    t_yt = np.zeros((M2, N2), dtype=complex)

    for i in np.arange(M2):
        for j in np.arange(N2):
            if i == M2 / 2 or j == N2 / 2:  # Nyquist frequency
                Gt = np.zeros((2, 2))
                Gt[0, 0] = 2 * (1 + nu) / (E * k[i, j] ** 3) * ((1 - nu) * k[i, j] ** 2 + nu * ky[i, j] ** 2)
                Gt[1, 1] = 2 * (1 + nu) / (E * k[i, j] ** 3) * ((1 - nu) * k[i, j] ** 2 + nu * kx[i, j] ** 2)

                a = (Gt.T * Gt + alpha * np.eye(2)) ** -1 * Gt.T
                a[np.isnan(a)] = 0
                b = (u_fft[i, j], v_fft[i, j])
                Tt = np.dot(a, b)
                t_xt[i, j] = Tt[0]
                t_yt[i, j] = Tt[1]

            elif ~((i == 1) and (j == 1)):
                Gt = np.zeros((2, 2))
                Gt[0, 0] = 2 * (1 + nu) / (E * k[i, j] ** 3) * ((1 - nu) * k[i, j] ** 2 + nu * ky[i, j] ** 2)
                Gt[1, 1] = 2 * (1 + nu) / (E * k[i, j] ** 3) * ((1 - nu) * k[i, j] ** 2 + nu * kx[i, j] ** 2)
                Gt[0, 1] = - nu * kx[i, j] * ky[i, j]
                Gt[1, 0] = - nu * kx[i, j] * ky[i, j]

                a = (Gt.T * Gt + alpha * np.eye(2)) ** -1 * Gt.T
                a[np.isnan(a)] = 0
                b = (u_fft[i, j], v_fft[i, j])
                Tt = np.dot(a, b)
                t_xt[i, j] = Tt[0]
                t_yt[i, j] = Tt[1]

    t_x = ifft2(t_xt)
    t_y = ifft2(t_yt)
    traction_x = np.real(t_x)
    traction_y = np.real(t_y)

    return traction_x[0:M, 0:N], traction_y[0:M, 0:N]





def main(savepath, pixelsize, downsamplerate=8, E=19960, nu=0.5, alpha=1*1e-19):


    pixelsize *= 1e-6
    forcemap_pixelsize = pixelsize * downsamplerate

    d_x = np.load(savepath + "d_x.npy") * pixelsize
    d_y = np.load(savepath + "d_y.npy") * pixelsize
    no_frames = d_x.shape[0]


    # calculate forces and make force plots
    t_x = np.zeros(d_x.shape)
    t_y = np.zeros(d_y.shape)
    for frame in range(no_frames):
        print("Calculating traction forces for frame: " + str(frame))
        t_x_current, t_y_current = calculate_traction_stresses(d_x[frame, :, :], d_y[frame, :, :], E, nu, forcemap_pixelsize, alpha)


        t_x[frame, :, :] = t_x_current
        t_y[frame, :, :] = t_y_current

    np.save(savepath + "t_x.npy", t_x)
    np.save(savepath + "t_y.npy", t_y)

