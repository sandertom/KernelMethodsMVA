"""
This implementation is inspired from the implementation of skimage's HOG implementation.
As you can see, the code is shorter from the original one, because we leave less choice to the user.
https://github.com/scikit-image/scikit-image/blob/main/skimage/feature/_hog.py#L48-L307
"""
import numpy as np

def ch_gradient(ch):
    #Computes the horizontal and vertical gradients of the image for one channel
    g_r, g_c = np.zeros(ch.shape),  np.zeros(ch.shape)
    g_r[1:-1, :] = ch[2:, :] - ch[:-2, :] #computes gradient for rows
    g_c[:, 1:-1] = ch[:, 2:] - ch[:, :-2] #computes gradient for columns
    return g_r, g_c

def cell_hog(norm,orientation,start_or, end_or, cell_c, cell_r,index_c, index_r, size_c, size_r, start_r, stop_r, start_c, stop_c):
    #Computes the weight of the gradient for a range of orientations in one cell
    total = 0.
    for cell_row in range(start_r, stop_r):
        cell_index_r = index_r + cell_row
        if (cell_index_r >= 0 and cell_index_r < size_r):
            for cell_column in range(start_c, stop_c):
                cell_index_c = index_c + cell_column
                if (cell_index_c >= 0 and cell_index_c < size_c and orientation[cell_index_r, cell_index_c] < start_or and orientation[cell_index_r, cell_index_c] > end_or):
                    total += norm[cell_index_r, cell_index_c]
    return total/(cell_r * cell_c)


def normalize_block_L2(block, eps=1e-10): #Normalizes each block with "L2-Hys"" regularization
    out = block/np.sqrt(np.sum(block**2)+eps)
    out = np.minimum(out, 0.2)
    out = out/np.sqrt(np.sum(out**2)+eps)
    return out

def histograms_hog(gradient_columns,gradient_rows,cell_c,cell_r,size_c, size_r,number_of_cells_c, number_of_cells_r, number_of_orientations, orientation_histogram):
    #computes the histogram of orientations for each cell
    norm = np.hypot(gradient_columns,gradient_rows)  #sqrt(x1**2 + x2**2) element-wise, norm of the gradients
    orientation = np.rad2deg(np.arctan2(gradient_rows, gradient_columns)) % 180 #orientations

    r0, c0, cc, cr= cell_r//2, cell_c//2, cell_r*number_of_cells_r, cell_c * number_of_cells_c
    stop_r, start_r, stop_c, start_c = int((cell_r + 1) / 2),int(-(cell_r / 2)), int((cell_c + 1) / 2), int(-(cell_c / 2))
    nbr_or = 180. / number_of_orientations

    for i in range(number_of_orientations):
        start_or = nbr_or*(i+1)
        end_or = nbr_or*i
        c,r,ri,ci = c0,r0,0,0
        while r < cc:
            ci,c = 0,c0
            while c < cr:
                orientation_histogram[ri, ci, i] = cell_hog(norm, orientation,start_or, end_or,cell_c, cell_r, c, r, size_c, size_r, start_r, stop_r,start_c, stop_c)
                ci += 1
                c += cell_c
            ri += 1
            r += cell_r

def hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3)):
    #Combines evertything and combines multichannel HOG
    g_r_by_ch, g_c_by_ch, g_magn = np.empty_like(image), np.empty_like(image), np.empty_like(image)

    for index_ch in range(3):
        g_r_by_ch[:, :, index_ch], g_c_by_ch[:, :, index_ch] = ch_gradient(image[:, :, index_ch])
        g_magn[:, :, index_ch] = np.hypot(g_r_by_ch[:, :, index_ch], g_c_by_ch[:, :, index_ch])

    idcs_max = g_magn.argmax(axis=2)
    rr, cc = np.meshgrid(np.arange(image.shape[0]),np.arange(image.shape[1]),indexing='ij',sparse=True)
    g_r, g_c = g_r_by_ch[rr, cc, idcs_max], g_c_by_ch[rr, cc, idcs_max]
    s_r, s_c = image.shape[:2]
    c_r, c_c = pixels_per_cell
    b_r, b_c = cells_per_block

    n_cells_r, n_cells_c= int(s_r // c_r), int(s_c // c_c)

    # compute orientations integral images
    orientation_histogram = np.zeros((n_cells_r, n_cells_c, orientations),dtype=float)
    g_r, g_c = g_r.astype(float, copy=False), g_c.astype(float, copy=False)
    histograms_hog(g_c, g_r, c_c, c_r, s_c, s_r,n_cells_c, n_cells_r, orientations, orientation_histogram)

    # now compute the histogram for each cell
    n_blocks_r, n_blocks_c= (n_cells_r - b_r) + 1, (n_cells_c - b_c) + 1
    normalized_blocks = np.zeros((n_blocks_r, n_blocks_c, b_r, b_c, orientations))
    for r in range(n_blocks_r):
        for c in range(n_blocks_c):
            block = orientation_histogram[r:r + b_r, c:c + b_c, :]
            normalized_blocks[r, c, :] = normalize_block_L2(block)
            
    return normalized_blocks.ravel()