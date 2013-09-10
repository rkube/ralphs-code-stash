#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

import numpy as np

# Machine coordinates of pixel view
r_arr = np.linspace(88.00, 91.08, 9)
z_arr = np.linspace(-4.51, -1.08, 10)

# Bad pixels, [Z, R]. See ~/Dropbox/matlab/acm_2d_combine
# From Odd Eriks code
#bad_px_list = [(0, 4),
#               (0, 5),
#               (0, 6),
#               (1, 4),
#               (1, 5),
#               (2, 0),
#               (2, 3),
#               (2, 4),
#               (2, 5),
#               (3, 3),
#               (3, 4),
#               (4, 3),
#               (5, 3),
#               (6, 5),
#               (7, 4),
#               (7, 5),
#               (8, 4),
#               (9, 4),
#               (9, 5)]


# Used apd_test_bad_pixels
# Bad px list, host 1111208020
bad_px_list = [(1, 0),
               (5, 0),
               (7, 0),
               (9, 0),
               (1, 1),
               (2, 1),
               (9, 1),
               (0, 2),
               (2, 3),
               (6, 3),
               (9, 3),
               (7, 4),
               (9, 6),
               (3, 7),
               (8, 7),
               (9, 7),
               (2, 8),
               (3, 8),
               (4, 8),
               (6, 8),
               (7, 8),
               (8, 8),
               (9, 8)]



# End of file apd_helper.py
