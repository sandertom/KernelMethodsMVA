"""
This implementation is inspired by the original implementation of skimage's HOG implementation.
https://github.com/scikit-image/scikit-image/blob/main/skimage/feature/_hog.py#L48-L307
"""


import numpy as np

def _hog_normalize_block(block, eps=1e-5):
    #Normalizes each block with L1 refularization
    out = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
    out = np.minimum(out, 0.2)
    out = out / np.sqrt(np.sum(out ** 2) + eps ** 2)
    return out

def _hog_channel_gradient(channel):
    #Computes the horizontal and vertical gradients of the image
    g_row = np.empty(channel.shape, dtype=channel.dtype)
    g_row[0, :] = 0
    g_row[-1, :] = 0
    g_row[1:-1, :] = channel[2:, :] - channel[:-2, :]
    g_col = np.empty(channel.shape, dtype=channel.dtype)
    g_col[:, 0] = 0
    g_col[:, -1] = 0
    g_col[:, 1:-1] = channel[:, 2:] - channel[:, :-2]
    return g_row, g_col

def cell_hog(magnitude,orientation,orientation_start, orientation_end, cell_columns, cell_rows,\
             column_index, row_index, size_columns, size_rows, range_rows_start, range_rows_stop,\
             range_columns_start, range_columns_stop):
    #Computes the weight of the gradient for a range of orientations in one cell
    total = 0.

    for cell_row in range(range_rows_start, range_rows_stop):
        cell_row_index = row_index + cell_row
        if (cell_row_index < 0 or cell_row_index >= size_rows):
            continue

        for cell_column in range(range_columns_start, range_columns_stop):
            cell_column_index = column_index + cell_column
            if (cell_column_index < 0 or cell_column_index >= size_columns
                    or orientation[cell_row_index, cell_column_index]
                    >= orientation_start
                    or orientation[cell_row_index, cell_column_index]
                    < orientation_end):
                continue

            total += magnitude[cell_row_index, cell_column_index]

    return total / (cell_rows * cell_columns)


def hog_histograms(gradient_columns,gradient_rows,cell_columns,cell_rows,size_columns, size_rows,\
                   number_of_cells_columns, number_of_cells_rows, number_of_orientations, orientation_histogram):
    #computes the histogram of orientations for each cell
    magnitude = np.hypot(gradient_columns,gradient_rows)
    orientation = np.rad2deg(np.arctan2(gradient_rows, gradient_columns)) % 180

    r_0 = cell_rows // 2
    c_0 = cell_columns // 2
    cc = cell_rows * number_of_cells_rows
    cr = cell_columns * number_of_cells_columns
    range_rows_stop = int((cell_rows + 1) / 2)
    range_rows_start = int(-(cell_rows / 2))
    range_columns_stop = int((cell_columns + 1) / 2)
    range_columns_start = int(-(cell_columns / 2))
    number_of_orientations_per_180 = 180. / number_of_orientations

    for i in range(number_of_orientations):
        orientation_start = number_of_orientations_per_180 * (i + 1)
        orientation_end = number_of_orientations_per_180 * i
        c = c_0
        r = r_0
        r_i = 0
        c_i = 0
        while r < cc:
            c_i = 0
            c = c_0
            while c < cr:
                orientation_histogram[r_i, c_i, i] = \
                    cell_hog(magnitude, orientation,
                             orientation_start, orientation_end,
                             cell_columns, cell_rows, c, r,
                             size_columns, size_rows,
                             range_rows_start, range_rows_stop,
                             range_columns_start, range_columns_stop)
                c_i += 1
                c += cell_columns

            r_i += 1
            r += cell_rows

def hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3)):
    #Combines evertything 
    g_row, g_col = _hog_channel_gradient(image)

    s_row, s_col = image.shape[:2]
    c_row, c_col = pixels_per_cell
    b_row, b_col = cells_per_block

    n_cells_row = int(s_row // c_row)  # number of cells along row-axis
    n_cells_col = int(s_col // c_col)  # number of cells along col-axis

    # compute orientations integral images
    orientation_histogram = np.zeros((n_cells_row, n_cells_col, orientations),
                                     dtype=float)
    g_row = g_row.astype(float, copy=False)
    g_col = g_col.astype(float, copy=False)

    hog_histograms(g_col, g_row, c_col, c_row, s_col, s_row,
                                 n_cells_col, n_cells_row,
                                 orientations, orientation_histogram)

    # now compute the histogram for each cell

    n_blocks_row = (n_cells_row - b_row) + 1
    n_blocks_col = (n_cells_col - b_col) + 1
    normalized_blocks = np.zeros((n_blocks_row, n_blocks_col, b_row, b_col, orientations))

    for r in range(n_blocks_row):
        for c in range(n_blocks_col):
            block = orientation_histogram[r:r + b_row, c:c + b_col, :]
            normalized_blocks[r, c, :] = \
                _hog_normalize_block(block)

    
    normalized_blocks = normalized_blocks.ravel()

    return normalized_blocks