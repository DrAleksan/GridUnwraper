import numpy as np
import cv2
import pickle
import cmath
import matplotlib.pyplot as plt

SIZEX = 800
SIZEY = 800

def go_horisontal(cell, number_to_cells, founded_mask, rest_cells):  # возвращает строку из клеток горизонтальных с данной
    row = [cell]
    dir_right = (cell.p4[0] - cell.p1[0], cell.p4[1] - cell.p1[1])
    dir_left = (cell.p1[0] - cell.p4[0], cell.p1[1] - cell.p4[1])

    print(f'in the begining {dir_right} {dir_left}')

    cur = cell

    new_center = (int(cur.center[0] + dir_left[0]), int(cur.center[1] + dir_left[1]))

    while 0 < new_center[0] < SIZEX and 0 < new_center[1] < SIZEY:
        print(f'aaaaaaa  {SIZEX} {SIZEY}')
        founded_number = founded_mask[new_center[1]][new_center[0]]
        if founded_number == 0:
            break

        if cur == number_to_cells[founded_number]:
            break

        cur = number_to_cells[founded_number]

        if cur not in rest_cells:
            break

        dir_left = (cur.p1[0] - cur.p4[0], cur.p1[1] - cur.p4[1])

        row = [cur] + row


        print(f'd_left {dir_left} {new_center}')

        new_center = (int(cur.center[0] + dir_left[0]), int(cur.center[1] + dir_left[1]))

    cur = cell

    new_center = (int(cur.center[0] + dir_right[0]), int(cur.center[1] + dir_right[1]))

    while 0 < new_center[0] < SIZEX and 0 < new_center[1] < SIZEY:
        print('bbbbbb')
        founded_number = founded_mask[new_center[1]][new_center[0]]
        if founded_number == 0:
            break

        if cur == number_to_cells[founded_number]:
            break

        cur = number_to_cells[founded_number]

        if cur not in rest_cells:
            break

        dir_right = (cur.p4[0] - cur.p1[0], cur.p4[1] - cur.p1[1])

        row.append(cur)

        new_center = (int(cur.center[0] + dir_right[0]), int(cur.center[1] + dir_right[1]))

    return row


def make_full(row, x, y, min, max):
    for i in range(min, x):
        row.insert(0, None)

    for i in range(y, max):
        row.append(None)


class KGrid:
    # def __init__(self, cells, number_to_cells, founded_mask):
    #     rest_cells = cells.copy()
    #     self.rows = []
    #     max_row = 0
    #     for j in range(0, 800):
    #         i = 0
    #         cur = number_to_cells[founded_mask[j][i]]
    #         while cur == 0:
    #             i += 10
    #             if i > 800:
    #                 return
    #             cur = number_to_cells[founded_mask[j][i]]
    #         if cur in rest_cells:
    #             row = go_horisontal(cur)
    #             self.rows = self.rows.append(row)
    #             j += cur.height
    #             if len(row) > max_row:
    #                 max_row = len(row)
    #             for elem in self.rows:
    #                 rest_cells.remove(rest_cells)
    #     self.rows_number = len(self.rows)
    #     self.column_number = max_row

    def __init__(self, cells, number_to_cells, founded_mask, sizex, sizey):
        global SIZEX
        global SIZEY
        SIZEY, SIZEX = sizey, sizex
        self.rows = []
        rest_cells = cells.copy()

        self.width = 0
        self.height = 0

        print(number_to_cells)

        for j in range(0, SIZEY):
            for i in range(0, SIZEX):
                print(f'{SIZEX} {SIZEY}')
                print(f'{j} {i} ??')
                cur_number = founded_mask[j][i]
                if cur_number == 0:
                    i += 5
                    continue

                cur_cell = number_to_cells[cur_number]

                if cur_cell in rest_cells:
                    row = go_horisontal(cur_cell, number_to_cells, founded_mask, rest_cells)
                    self.rows.append(row)

                    for elem in row:
                        if elem in rest_cells:
                            rest_cells.remove(elem)

        mean_length = np.mean([len(row) for row in self.rows])

        self.rows = list(filter(lambda x: len(x) > mean_length*.9, self.rows))

        self.rows.sort(key = lambda x: x[0].p1[1])

        self.connect_vertical(number_to_cells, founded_mask)

    def connect_vertical(self, number_to_cells, founded_mask):
        start_end = {}

        top_bottom = {}

        start = 0

        length = len(self.rows)

        print(length)
        #
        # next_to_find = [self.rows[0]]
        #
        # while len(next_to_find) != 0:
        #     cur = next_to_find.pop(0)
        #
        #     for i, row in enumerate(cur):
        #         end = start + len(row)
        #         start_end[i] = (start, end)
        #
        #         if i == length-1:
        #             break
        #
        #         for j, cell in enumerate(cur)

        for i, row in enumerate(self.rows):
            end = start + len(row)
            start_end[i] = (start, end)

            if i == length-1:
                break

            for j, cell in enumerate(row):
                dir_down = (cell.p2[0] - cell.p1[0], cell.p2[1] - cell.p1[1])

                new_center = (int(cell.center[0] + dir_down[0]), int(cell.center[1] + dir_down[1]))
                if 0 < new_center[0] < 800 and 0 < new_center[1] < 800:
                    founded_number = founded_mask[new_center[1]][new_center[0]] # не рассмотрен случай с пустой таблицей, должно быть исправленно в cell
                    if founded_number == 0:
                        continue
                    # print(i)

                    founded = False

                    key = 0

                    print(f'start: {start} i: {i}  j: {j}')

                    for k, r in enumerate(self.rows):
                        if number_to_cells[founded_number] in r:
                            founded = True
                            key = k

                    if founded:
                        # print(f'i: {i}  j : {j}')
                        start = start + j - self.rows[key].index(number_to_cells[founded_number])
                        break

                    # if number_to_cells[founded_number] in self.rows[i+1]:
                    #     start = start + j - self.rows[key].index(number_to_cells[founded_number])

        min = None
        max = 0

        for _, v in start_end.items():
            if min == None:
                min = v[0]
            if min > v[0]:
                min = v[0]
            if max < v[1]:
                max = v[1]

        # print(f'min {min}   max {max}')

        for i, row in enumerate(self.rows):
            # print(i)
            make_full(row, start_end[i][0], start_end[i][1], min, max)
        #
        # for row in self.rows:
        #     print(f'length - {len(row)}')
        #
        # print(start_end)
        #
        # for row in self.rows:
        #     print(row)

        self.width = max - min
        self.height = length

    def get_centers(self):
        result = []
        for row in self.rows:
            row_of_centers = []
            for elem in row:
                if elem == None:
                    continue
                row_of_centers.append(elem.center)
            result.append(row_of_centers)
        return result

    def get_upper_right_points(self): # проблемы т.к не известно какая точка является правой верхней
        result = []
        for row in self.rows:
            row_of_centers = []
            for elem in row:
                row_of_centers = row_of_centers.append(elem.center)
            result = result.append(row_of_centers)
        return result

    def unwrap(self, image):
        result = np.zeros((self.height*20, self.width*20, 3), np.uint8)

        for j, row in enumerate(self.rows):
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
    # def unwrap(self, points, image):
    #     height = len(points)
    #     width = len(points[0])
    #
    #     result = np.zeros((height*20, width*20, 3))
    #
    #     for j in range(height-1):
    #         for i in range(width-1):
    #             pts1 = np.float32([points[j][i], points[j][i+1], points[j+1][i], points[j+1][i+1]])
    #             pts2 = np.float32([[0, 20], [20, 20], [0, 0], [20, 0]])
    #
    #             m = cv2.getPerspectiveTransform(pts1, pts2)
    #             dst = cv2.warpPerspective(image, m, (20, 20))
    #
    #             result[j*20:(j+1)*20, i*20:(i+1)*20] = dst
    #     return result


