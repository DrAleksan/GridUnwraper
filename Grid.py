import numpy as np
import cv2
import pickle
import cmath
import matplotlib.pyplot as plt

import cell

SIZEX = 800
SIZEY = 800


def make_full(row, x, y, min, max):
    for i in range(min, x):
        row.insert(0, None)

    for i in range(y, max):
        row.append(None)


class KGrid:
    def __init__(self, cells, number_to_cells, founded_mask, sizex, sizey):
        global SIZEX
        global SIZEY
        SIZEY, SIZEX = sizey, sizex
        self.cells = cells.copy()

        self.connected_cells = np.ndarray(shape = (1, 1), dtype = cell.Cell)

        self.connected_cells[0, 0] = self.cells[0]

        self.connect_cells(founded_mask)

    def connect_cells(self, founded_mask):
        rest_cells = [(self.connected_cells[0, 0], (0, 0))]

        x_bias, y_bias = 0, 0

        while len(rest_cells) != 0:
            cur, (y, x) = rest_cells.pop(0)

            x = x + x_bias
            y = y + y_bias

            leny, lenx = self.connected_cells.shape

            print(f'leny: {leny} lenx: {lenx}')
            print(self.connected_cells)

            if y == leny-1:
                self.connected_cells = np.insert(self.connected_cells, y+1, [None]*lenx, 0)
                leny += 1

            if y == 0:
                self.connected_cells = np.insert(self.connected_cells, 0, [None]*lenx, 0)
                leny += 1
                y += 1
                y_bias += 1

            if x == lenx-1:
                self.connected_cells = np.insert(self.connected_cells, x+1, [None]*leny, 1)
                lenx += 1

            if x == 0:
                self.connected_cells = np.insert(self.connected_cells, 0, [None]*leny, 1)
                lenx += 1
                x += 1
                x_bias += 1

            if self.connected_cells[y+1, x] is None:
                bottom = cur.get_bottom_neighborhood(founded_mask)

                if not (bottom is None):
                    rest_cells.append((bottom, (y + 1 - y_bias, x - x_bias)))
                    self.connected_cells[y + 1, x] = bottom

            if self.connected_cells[y - 1, x] is None:
                top = cur.get_upper_neighborhood(founded_mask)
                if not (top is None):
                    rest_cells.append((top, (y - 1 - y_bias, x - x_bias)))
                    self.connected_cells[y - 1, x] = top

            if self.connected_cells[y, x + 1] is None:
                right = cur.get_right_neighborhood(founded_mask)
                if not (right is None):
                    rest_cells.append((right, (y - y_bias, x + 1 - x_bias)))
                    self.connected_cells[y, x + 1] = right

            if self.connected_cells[y, x - 1] is None:
                left = cur.get_left_neighborhood(founded_mask)
                if not (left is None):
                    rest_cells.append((left, (y - y_bias, x - 1 - x_bias)))
                    self.connected_cells[y, x - 1] = left


        self.height, self.width = self.connected_cells.shape

    def get_centers(self):
        result = []
        for row in self.connected_cells:
            row_of_centers = []
            for elem in row:
                if elem == None:
                    continue
                row_of_centers.append(elem.center)
            result.append(row_of_centers)
        return result

    def get_upper_right_points(self):
        result = []
        for row in self.rows:
            row_of_centers = []
            for elem in row:
                row_of_centers = row_of_centers.append(elem.center)
            result = result.append(row_of_centers)
        return result

    def unwrap(self, image):
        result = np.zeros((self.height*20, self.width*20, 3), np.uint8)

        for j, row in enumerate(self.connected_cells):
            for i, elem in enumerate(row):
                dst = None
                if elem == None:
                    dst = np.zeros((20, 20, 3))
                else:
                    pts1 = np.float32([elem.p1, elem.p2, elem.p3, elem.p4])
                    pts2 = np.float32([[0, 0], [0, 20], [20, 20], [20, 0]])

                    m = cv2.getPerspectiveTransform(pts1, pts2)
                    dst = cv2.warpPerspective(image, m, (20, 20))


                result[j*20:(j+1)*20, i*20:(i+1)*20] = dst
        return result

