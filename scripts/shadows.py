import matplotlib.pyplot as plt
import numpy as np

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def main():
    dim = 100
    chem_x = np.zeros((dim, dim))
    chem_y = np.zeros((dim, dim))

    for i in range(dim):
        for j in range(dim):
            if np.abs(i - j) < 5:
                chem_y[i, j] = 1
            else:
                chem_x[i, j] = 1

    light = np.array([0, dim, 10])
    height = 1
    resolution = np.array([dim, dim])
    lighting = np.zeros((dim, dim, 3))

    colourA = np.array([1, 0, 0])
    colourB = np.array([0, 1, 0])
    out_color = np.zeros((dim, dim, 3))

    for i in range(dim):
        for j in range(dim):
            lighting[i,j,:] = np.array([1, 1, 1])
            coord = np.array([i, j])
            pos = coord
            pos_height = chem_y[i,j] * height
            dist_to_light = np.linalg.norm(coord - light[:2])
            max_dist_to_other_peak = dist_to_light * (height - pos_height) / (light[2] - pos_height)
            dir_to_light = normalize(light[:2] - pos)
            for dist in np.arange(0, max_dist_to_other_peak, 1):
                other_peak = pos + (dir_to_light * dist)
                if (other_peak[0] < 0 or other_peak[0] >= resolution[0] or other_peak[1] < 0 or other_peak[1] >= resolution[1]):
                    break

                other_peak_height = chem_y[int(other_peak[0]), int(other_peak[1])] * height
                light_height = pos_height + (dist * (light[2] - pos_height) / dist_to_light)
                if (other_peak_height > light_height):
                    # in shadow
                    lighting[i,j,:] = np.array([0.5, 0.5, 0.5])
                    break
            lighting[i,j,:] *= np.exp(-dist_to_light / 200.)

            colour = chem_x[i,j] * colourA * lighting[i,j]
            colour += chem_y[i,j] * colourB * lighting[i,j]
            out_color[i,j,:] = colour

    plt.imshow(out_color)
    plt.show()

if __name__ == '__main__':
    main()