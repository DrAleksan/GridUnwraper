import numpy as np
import cv2
import pickle
import cmath
import matplotlib.pyplot as plt
from Grid import KGrid
import random


NUMBER_OF_CELL = 1  # Номер текущей клетки
NUMBER_TO_DICT = {}  # Map, отображающий номер клетки в клетку

POINT_TO_CELL = {}  # Map, отображающий точку в пару (клетка, номер точки)

LINE_WIDTH = 2

D = 5

SIZEX = 800
SIZEY = 800

thresholded = None

class Cell:


    def __init__(self, image, mask, founded_mask, build=True, main=False):
        global SIZEX
        global SIZEY
        SIZEY, SIZEX, _ = image.shape
        self.main = main
        self.image = image
        self.mask = mask # маска в которой контур чёрный а внутренность белая

        self.p1, self.p2, self.p3, self.p4, self.successful = 0, 0, 0, 0, True
        self.d1, self.d2, self.d3, self.d4 = 0, 0, 0, 0
        self.center = [0, 0]
        self.len_h = 0
        self.len_w = 0

        self.solid = [False, False, False, False]

        if build:
            self.calc_all()

            if not self.successful:
                return

            NUMBER_TO_DICT[NUMBER_OF_CELL] = self
            color_square_and_increase_number(founded_mask, [self.p1, self.p2, self.p3, self.p4])

    def calc_all(self):
        self.p1, self.p2, self.p3, self.p4, self.successful = self.find_points(self.mask)

        if not self.successful:
            return

        # cv2.circle(self.image, (self.p1[0], self.p1[1]), 3, (255, 255, 255), -1)
        # cv2.circle(self.image, (self.p2[0], self.p2[1]), 3, (255, 255, 255), -1)
        # cv2.circle(self.image, (self.p3[0], self.p3[1]), 3, (255, 255, 255), -1)
        # cv2.circle(self.image, (self.p4[0], self.p4[1]), 3, (255, 255, 255), -1)

        self.d1, self.d2, self.d3, self.d4 = self.find_directions(self.p1, self.p2,
                                                                  self.p3, self.p4)

        if not self.successful:
            return

        self.bind_points(self.p1, self.p2, self.p3, self.p4)

        all_p_x = [self.p1[0], self.p2[0], self.p3[0], self.p4[0]]
        all_p_y = [self.p1[1], self.p2[1], self.p3[1], self.p4[1]]

        self.center = [int(np.mean(all_p_x)), int(np.mean(all_p_y))]

    def bind_points(self, p1, p2, p3, p4):  # не рассмотрен случай пересечениея квадратов, но не попадания центров в квадраты
        x, y = p1

        founded = False

        for j in range(y-D, y+D):
            for i in range(x-D, x+D):
                if (j, i) in POINT_TO_CELL:
                    founded = True
                    self.adjust_point(i, j, 1)
                    break
            if founded:
                break

        if not founded:
            asign_area(self, x, y, 1)

        x, y = p2

        founded = False

        for j in range(y - D, y + D):
            for i in range(x - D, x + D):
                if (j, i) in POINT_TO_CELL:
                    founded = True
                    self.adjust_point(i, j, 2)
                    break
            if founded:
                break

        if not founded:
            asign_area(self, x, y, 2)

        x, y = p3

        founded = False

        for j in range(y - D, y + D):
            for i in range(x - D, x + D):
                if (j, i) in POINT_TO_CELL:
                    founded = True
                    self.adjust_point(i, j, 3)
                    break
            if founded:
                break

        if not founded:
            asign_area(self, x, y, 3)

        x, y = p4

        founded = False

        for j in range(y - D, y + D):
            for i in range(x - D, x + D):
                if (j, i) in POINT_TO_CELL:
                    founded = True
                    self.adjust_point(i, j, 4)
                    break
            if founded:
                break

        if not founded:
            asign_area(self, x, y, 4)

    @classmethod
    def create_from_cell(cls, image, width, height, mask, center, points, directions, founded_mask, thresholded):
        new_cell = cls(image, width, height, mask, founded_mask, thresholded, build=False)
        new_cell.mask = mask
        new_cell.has_mask = True
        new_cell.p1, new_cell.p2, new_cell.p3, new_cell.p4 = points
        new_cell.d1, new_cell.d2, new_cell.d3, new_cell.d4 = directions
        new_cell.center = [center[1], center[0]]
        NUMBER_TO_DICT[NUMBER_OF_CELL] = new_cell
        color_square_and_increase_number(founded_mask, [new_cell.p1, new_cell.p2, new_cell.p3, new_cell.p4])
        return new_cell

    @staticmethod
    def find_points(segment):  # p1 - левая верхняя точка
                               # p2 - левая нижняя точка
                               # p3 - правая нижняя точка
                               # p4 - правая верхняя точка
        cnts = cv2.findContours(np.uint8(segment), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

        c = cnts[1][0]

        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, perimeter * .1, True)

        if len(approx) != 4:
            return 0, 0, 0, 0, False

        p1, p2, p3, p4 = approx[0][0], approx[1][0], approx[2][0], approx[3][0]

        points = [p1, p2, p3, p4]

        two_top_index = sorted(range(len(points)), key=lambda i: points[i][1], reverse=True)[-2:]

        two_left_index = sorted(range(len(points)), key=lambda i: points[i][0], reverse=True)[-2:]


        intersection = set(two_left_index) & set(two_top_index)

        if len(intersection) != 0:
            for i in range(list(intersection)[0]):
                p1, p2, p3, p4 = p2, p3, p4, p1

        p1 = (p1[0] - LINE_WIDTH, p1[1] - LINE_WIDTH)
        p2 = (p2[0] - LINE_WIDTH, p2[1] + LINE_WIDTH)
        p3 = (p3[0] + LINE_WIDTH, p3[1] + LINE_WIDTH)
        p4 = (p4[0] + LINE_WIDTH, p4[1] - LINE_WIDTH)

        return p1, p2, p3, p4, True

    def find_directions(self, p1, p2, p3, p4):
        # d1 - направление вверх
        # d2 - направление вниз
        # d3 - направление вправо
        # d4 - направление влево


        # first_direction1 = (p1, p2)
        # first_direction2 = (p3, p4)
        # intersect = self.find_intersect(first_direction1, first_direction2)
        # if 1 >= intersect >= -1: # В эту ветвь ничего не должно попадать
        #     print("problems")
        #     first_direction1 = (p1, p3)
        #     first_direction2 = (p2, p4)
        #     second_direction1 = (p1, p4)
        #     second_direction2 = (p2, p3)
        # else:
        #     second_direction1 = (p1, p3)
        #     second_direction2 = (p2, p4)
        #     intersect = self.find_intersect(second_direction1, second_direction2)
        #     if 1 >= intersect >= -1:
        #         second_direction1 = (p1, p4)
        #         second_direction2 = (p2, p3)
        #     else: # Сюда видимо тоже
        #         print("Problems")

        first_direction1 = (p1, p4)
        first_direction2 = (p2, p3)

        second_direction1 = (p1, p2)
        second_direction2 = (p4, p3)


        v11 = (first_direction1[0][0] - first_direction1[1][0], first_direction1[0][1] - first_direction1[1][1])
        v12 = (first_direction2[0][0] - first_direction2[1][0], first_direction2[0][1] - first_direction2[1][1])
        v21 = (second_direction1[0][0] - second_direction1[1][0], second_direction1[0][1] - second_direction1[1][1])
        v22 = (second_direction2[0][0] - second_direction2[1][0], second_direction2[0][1] - second_direction2[1][1])

        dist1 = cmath.sqrt((first_direction1[0][0] - first_direction1[1][0]) ** 2 + (
                first_direction1[0][1] - first_direction1[1][1]) ** 2).real
        dist2 = cmath.sqrt((first_direction2[0][0] - first_direction2[1][0]) ** 2 + (
                first_direction2[0][1] - first_direction2[1][1]) ** 2).real

        dist3 = cmath.sqrt((second_direction1[0][0] - second_direction1[1][0]) ** 2 + (
                    second_direction1[0][1] - second_direction1[1][1]) ** 2).real
        dist4 = cmath.sqrt((second_direction2[0][0] - second_direction2[1][0]) ** 2 + (
                    second_direction2[0][1] - second_direction2[1][1]) ** 2).real

        if abs(dist1 - dist2) > 2 or abs(dist3 - dist4) > 2: # обобщить
            self.successful = False
            return 0, 0, 0, 0

        self.len_h = dist3
        self.len_w = dist1

#        print(dist1/dist3)

        if dist1/dist3 > 2 or dist1/dist3 < .6: #переделать на орентировку на стреднее значение
            self.successful = False
            return 0, 0, 0, 0

        if abs(self.calculatte_angle(v11, v12)) < 0.99 or abs(self.calculatte_angle(v21, v22)) < 0.99:
            self.successful = False
            return 0, 0, 0, 0

        n11 = self.get_ortogonal(v11)
        n12 = self.get_ortogonal(v12)
        n21 = self.get_ortogonal(v21)
        n22 = self.get_ortogonal(v22)

        if self.dot(n11, (0, 1)) < 0:
            n11 = (-n11[0], -n11[1])

        if self.dot(n12, (0, -1)) < 0:
            n12 = (-n12[0], -n12[1])

        if self.dot(n21, (-1, 0)) < 0:
            n21 = (-n21[0], -n21[1])

        if self.dot(n22, (1, 0)) < 0:
            n22 = (-n22[0], -n22[1])

        r11 = self.sum_and_normalize(n11, n12)
        r12 = (-r11[0], -r11[1])
        r21 = self.sum_and_normalize(n21, n22)
        r22 = (-r21[0], -r21[1])

        return r11, r12, r21, r22

    @staticmethod
    def get_ortogonal(v):
        x, y = v
        return y, -x

    def calculatte_angle(self, v, u):
        dot = self.dot(v, u)
        n1 = cmath.sqrt(self.dot(v, v)).real
        n2 = cmath.sqrt(self.dot(u, u)).real
        return dot / (n1 * n2)

    @staticmethod
    def dot(v1, v2):
        return v1[0] * v2[0] + v1[1] * v2[1]

    def sum_and_normalize(self, v1, v2):
        v1 = list(v1)
        v2 = list(v2)
        dot = self.dot(v1, v2)
        if dot < 0:
            v1[0] = -v1[0]
            v1[1] = -v1[1]
        result = (v1[0] + v2[0], v1[1] + v2[1])
        value = cmath.sqrt(self.dot(result, result)).real
        result = (result[0] / value, result[1] / value)
        return result

    @staticmethod
    def find_intersect(ps1, ps2):
        (x1, y1), (x2, y2) = ps1
        (x3, y3), (x4, y4) = ps2

        numerator = (-x4 * y1 + x2 * y1 - y2 * x1 + y2 * x2 + y2 * x4 - y2 * x2 + y4 * x1 - y4 * x2)
        denominator = (x3 * y1 - x4 * y1 - x3 * y2 + x4 * y2 - y3 * x1 + y3 * x2 + y4 * x1 - y4 * x2)

        if denominator != 0:
            return numerator / \
                   denominator
        else:
            return 1000

    def adjust_point(self, x, y, number):
        cell, n = POINT_TO_CELL[(y, x)][0]
        point = cell.get_point(n)

        new_x = int((point[0] + self.get_point(number)[0]) / 2)
        new_y = int((point[1] + self.get_point(number)[1]) / 2)

        POINT_TO_CELL[(y, x)].append((self, number))


        value = POINT_TO_CELL[(y, x)]

        for elem in POINT_TO_CELL[(y, x)]:
            elem[0].set_point(new_x, new_y, elem[1])
        clean_POINT_TO_CELL((new_x, new_y))

        for j in range(new_y - D, new_y + D):
            for i in range(new_x - D, new_x + D):
                POINT_TO_CELL[(j, i)] = value


    def draw_cell(self):
        all_p_x = [self.p1[0], self.p2[0], self.p3[0], self.p4[0]]
        all_p_y = [self.p1[1], self.p2[1], self.p3[1], self.p4[1]]
        origin = [np.mean(all_p_x)], [np.mean(all_p_y)]

        self.mask = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)

        cv2.line(self.mask, (int(origin[0][0]), int(origin[1][0])),
                 (int(origin[0][0] + self.d1[0] * 100), int(origin[1][0] + self.d1[1] * 100)), (255, 255, 0), 1)
        cv2.line(self.mask, (int(origin[0][0]), int(origin[1][0])),
                 (int(origin[0][0] + self.d2[0] * 100), int(origin[1][0] + self.d2[1] * 100)), (255, 255, 0), 1)
        cv2.line(self.mask, (int(origin[0][0]), int(origin[1][0])),
                 (int(origin[0][0] + self.d3[0] * 100), int(origin[1][0] + self.d3[1] * 100)), (255, 255, 0), 1)
        cv2.line(self.mask, (int(origin[0][0]), int(origin[1][0])),
                 (int(origin[0][0] + self.d4[0] * 100), int(origin[1][0] + self.d4[1] * 100)), (255, 255, 0), 1)
        cv2.imshow("current", self.mask)

        plt.scatter(all_p_x, all_p_y, s=10)

        # Set chart title.
        plt.title("Extract Number Root ")

        # Set x, y label text.
        plt.xlabel("Number")
        plt.ylabel("Extract Root of Number")

        xs = []
        ys = []

        for elem in [self.d1, self.d2, self.d3, self.d4]:
            xs.append(elem[0])
            ys.append(elem[1])

        plt.quiver(*origin, xs, ys, color=['r', 'b', 'g', 'm'], scale=21)
        plt.show()

    def find_nearby_cells(self, founded_mask, thresholded):  # проблемы с несоединяющимися точками!!!!!!!!!!!!!!!!!!!
        dir1 = (self.p1[0] - self.p2[0], self.p1[1] - self.p2[1])
        dir2 = (self.p2[0] - self.p1[0], self.p2[1] - self.p1[1])

        dir3 = (self.p4[0] - self.p1[0], self.p4[1] - self.p1[1])
        dir4 = (self.p1[0] - self.p4[0], self.p1[1] - self.p4[1])

        c1 = (int(self.center[0] + dir1[0]), int(self.center[1] + dir1[1]))
        c2 = (int(self.center[0] + dir2[0]), int(self.center[1] + dir2[1]))
        c3 = (int(self.center[0] + dir3[0]), int(self.center[1] + dir3[1]))
        c4 = (int(self.center[0] + dir4[0]), int(self.center[1] + dir4[1]))

        new_upper_cell = None
        new_bottom_cell = None

        result = []

        if 0 < c1[0] < SIZEX and 0 < c1[1] < SIZEY and founded_mask[c1[1]][c1[0]] == 0:
            # cv2.circle(self.image, (c1[0], c1[1]), 3, (255, 255, 255), -1)
            new_upper_cell = Cell(self.image, self.mask,
                              founded_mask, main=False, build=False)

            new_upper_cell.p2 = self.p1
            new_upper_cell.p3 = self.p4
            new_upper_cell.p1 = (int(new_upper_cell.p2[0] + dir1[0]), int(new_upper_cell.p2[1] + dir1[1]))
            new_upper_cell.p4 = (int(new_upper_cell.p3[0] + dir1[0]), int(new_upper_cell.p3[1] + dir1[1]))

            new_upper_cell.d1 = self.d1
            new_upper_cell.d2 = self.d2
            new_upper_cell.d3 = self.d3
            new_upper_cell.d4 = self.d4

            new_upper_cell.center = c1
            new_upper_cell.len_h = self.len_h
            new_upper_cell.len_w = self.len_w

            new_upper_cell.bind_points(new_upper_cell.p1, new_upper_cell.p2, new_upper_cell.p3, new_upper_cell.p4)

            NUMBER_TO_DICT[NUMBER_OF_CELL] = new_upper_cell
            color_square_and_increase_number(founded_mask, [new_upper_cell.p1, new_upper_cell.p2,
                                                            new_upper_cell.p3, new_upper_cell.p4])

            result.append(new_upper_cell)

        if 0 < c2[0] < SIZEX and 0 < c2[1] < SIZEY and founded_mask[c2[1]][c2[0]] == 0:
            # cv2.circle(self.image, (c1[0], c1[1]), 3, (255, 255, 255), -1)
            new_bottom_cell = Cell(self.image, self.mask,
                              founded_mask, main=False, build=False)

            new_bottom_cell.p1 = self.p2
            new_bottom_cell.p4 = self.p3
            new_bottom_cell.p2 = (int(new_bottom_cell.p1[0] + dir2[0]), int(new_bottom_cell.p1[1] + dir2[1]))
            new_bottom_cell.p3 = (int(new_bottom_cell.p4[0] + dir2[0]), int(new_bottom_cell.p4[1] + dir2[1]))

            new_bottom_cell.d1 = self.d1
            new_bottom_cell.d2 = self.d2
            new_bottom_cell.d3 = self.d3
            new_bottom_cell.d4 = self.d4

            new_bottom_cell.center = c2
            new_bottom_cell.len_h = self.len_h
            new_bottom_cell.len_w = self.len_w

            new_bottom_cell.bind_points(new_bottom_cell.p1, new_bottom_cell.p2, new_bottom_cell.p3, new_bottom_cell.p4)

            NUMBER_TO_DICT[NUMBER_OF_CELL] = new_bottom_cell
            color_square_and_increase_number(founded_mask, [new_bottom_cell.p1, new_bottom_cell.p2,
                                                            new_bottom_cell.p3, new_bottom_cell.p4])

            result.append(new_bottom_cell)


        if 0 < c3[0] < SIZEX and 0 < c3[1] < SIZEY and founded_mask[c3[1]][c3[0]] == 0:
            # cv2.circle(self.image, (c1[0], c1[1]), 3, (255, 255, 255), -1)
            new_right_cell = Cell(self.image, self.mask,
                              founded_mask, main=False, build=False)

            new_right_cell.p1 = self.p4
            new_right_cell.p2 = self.p3
            new_right_cell.p3 = (int(new_right_cell.p2[0] + dir3[0]), int(new_right_cell.p2[1] + dir3[1]))
            new_right_cell.p4 = (int(new_right_cell.p1[0] + dir3[0]), int(new_right_cell.p1[1] + dir3[1]))

            new_right_cell.d1 = self.d1
            new_right_cell.d2 = self.d2
            new_right_cell.d3 = self.d3
            new_right_cell.d4 = self.d4

            new_right_cell.center = c3
            new_right_cell.len_h = self.len_h
            new_right_cell.len_w = self.len_w

            # print(f'{} {} {} {}')

            new_right_cell.bind_points(new_right_cell.p1, new_right_cell.p2, new_right_cell.p3, new_right_cell.p4)

            NUMBER_TO_DICT[NUMBER_OF_CELL] = new_right_cell
            color_square_and_increase_number(founded_mask, [new_right_cell.p1, new_right_cell.p2,
                                                            new_right_cell.p3, new_right_cell.p4])

            result.append(new_right_cell)

        if 0 < c4[0] < SIZEX and 0 < c4[1] < SIZEY and founded_mask[c4[1]][c4[0]] == 0:
            # cv2.circle(self.image, (c1[0], c1[1]), 3, (255, 255, 255), -1)
            new_left_cell = Cell(self.image, self.mask,
                              founded_mask, main=False, build=False)

            new_left_cell.p4 = self.p1
            new_left_cell.p3 = self.p2
            new_left_cell.p1 = (int(new_left_cell.p4[0] + dir4[0]), int(new_left_cell.p4[1] + dir4[1]))
            new_left_cell.p2 = (int(new_left_cell.p3[0] + dir4[0]), int(new_left_cell.p3[1] + dir4[1]))

            new_left_cell.d1 = self.d1
            new_left_cell.d2 = self.d2
            new_left_cell.d3 = self.d3
            new_left_cell.d4 = self.d4

            new_left_cell.center = c4
            new_left_cell.len_h = self.len_h
            new_left_cell.len_w = self.len_w

            # print(f'{} {} {} {}')

            new_left_cell.bind_points(new_left_cell.p1, new_left_cell.p2, new_left_cell.p3, new_left_cell.p4)

            NUMBER_TO_DICT[NUMBER_OF_CELL] = new_left_cell
            color_square_and_increase_number(founded_mask, [new_left_cell.p1, new_left_cell.p2,
                                                            new_left_cell.p3, new_left_cell.p4])

            result.append(new_left_cell)

        return result


    def draw_points(self, image):
        cv2.circle(image, (int(self.p1[0]), int(self.p1[1])), 1, (255, 255, 0), -1)
        cv2.circle(image, (int(self.p2[0]), int(self.p2[1])), 1, (0, 255, 255), -1)
        cv2.circle(image, (int(self.p3[0]), int(self.p3[1])), 1, (0, 255, 0), -1)
        cv2.circle(image, (int(self.p4[0]), int(self.p4[1])), 1, (0, 0, 255), -1)

    def get_point(self, number):

        if number == 1:
            return self.p1
        if number == 2:
            return self.p2
        if number == 3:
            return self.p3
        if number == 4:
            return self.p4

    def set_point(self, x, y, number):
        if number == 1:
            self.p1 = (x ,y)
        if number == 2:
            self.p2 = (x ,y)
        if number == 3:
            self.p3 = (x ,y)
        if number == 4:
            self.p4 = (x ,y)

    def get_number_to_dict(self):
        return NUMBER_TO_DICT

    # def count_inside_cell(self, number):
    #     result = 0
    #
    #     for j in range(SIZEY):
    #         for i in range(SIZEX):
    #             result +=

def asign_area(cell, x, y, number):
    for j in range(y-D, y+D):
        for i in range(x-D, x+D):
            POINT_TO_CELL[(j, i)] = [(cell, number)]


def color_square_and_increase_number(binary, points):
    global NUMBER_OF_CELL
    global NUMBER_TO_DICT

    points = np.array(points, 'int32')

    cv2.fillConvexPoly(binary, points, NUMBER_OF_CELL)

    NUMBER_OF_CELL = NUMBER_OF_CELL + 1


def points_to_cell_to_image():
    global POINT_TO_CELL

    image = np.zeros((800, 800), np.uint8)

    for j in range(0, 800):
        for i in range(0, 800):
            if (j, i) in POINT_TO_CELL:
                for _ in POINT_TO_CELL[(j, i)]:
                    image[j][i] += 30
    return image

# def unite_points():
#
#     checked = np.ones((800, 800))
#
#     for j in range(0, 800):
#         for i in range(0, 800):
#             if checked[j][i] == 0:
#                 continue
#             if (j, i) in POINT_TO_CELL:
#                 x = y = 0
#                 length = len(POINT_TO_CELL[(j, i)])
#
#                 if length == 1:
#                     continue
#
#
#                 for elem in POINT_TO_CELL[(j, i)]:
#                     cur_point = elem[0].get_point(elem[1])
#                     x += cur_point[0]
#                     y += cur_point[1]
#                 x = int(x/length)
#                 y = int(y/length)
#
#                 for elem in POINT_TO_CELL[(j, i)]:
#                     elem[0].set_point(x, y, elem[1])
#
#                 new_value = POINT_TO_CELL[(j, i)]
#
#
#
#                 for k in range(y-2*D, y+2*D):
#                     for l in range(x - 2*D, x+2*D):
# #                        POINT_TO_CELL[(k, l)] = new_value
#                         if (k, l) in POINT_TO_CELL:
#                             if k < 800 and l < 800 and k > 0 and l > 0:
#                                 checked[k][l] = 0
#                             del POINT_TO_CELL[(k, l)]
#
#                 for k in range(y-D, y+D):
#                     for l in range(x-D, x+D):
#                         POINT_TO_CELL[(k, l)] = new_value
#                 # POINT_TO_CELL[(y, x)] = new_value
#                 # POINT_TO_CELL[(y, x+1)] = new_value
#                 # POINT_TO_CELL[(y, x-1)] = new_value
#                 # POINT_TO_CELL[(y+1, x)] = new_value
#                 # POINT_TO_CELL[(y -1, x)] = new_value
#                 i += 4*D
#     print("ended")

def clean_POINT_TO_CELL(point):
    for j in range(point[1] - 2*D, point[1] + 2*D):
        for i in range(point[0] - 2*D, point[0] + 2*D):
            if (j, i) in POINT_TO_CELL:
                del POINT_TO_CELL[(j, i)]


#        roi = self.image[int(c1[0][0] - change - 10):int(c1[0][0] + change + 10), int(c1[1][0] - change - 10):int(c1[1][0] + change + 10)]

'''

with open('saved_variables_t2', 'rb') as f:
    image, blured_at, all_delete, repaired, output, only_squares, left_top_coords, mean_width, mean_height, final_mask = pickle.load(
        f)

# cv2.waitKey(0)

founded_mask = np.zeros((800, 800))

cells = []


for_experiments = image.copy()

for elem in only_squares:

    segment = output[1] == elem[5]
    segment = np.multiply(segment, 255)
    segment = np.uint8(segment)

    c = Cell(for_experiments, elem[cv2.CC_STAT_WIDTH], elem[cv2.CC_STAT_HEIGHT], segment, founded_mask, blured_at, main=True)
    if c.successful:
        cells.append(c)

# cv2.imshow("founded", founded_mask)
# cv2.imshow("new cell before", for_experiments)
#
# cv2.imshow("founded points", points_to_cell_to_image())
#
# cv2.waitKey(0)


cs = []

# for cell in cells:    !!!!!!!!!!!! infinite loop
#     result = cell.find_nearby_cells(founded_mask, blured_at)
#     if result is not None:
#         cells.extend(result)
#         cs.extend(result)
#
#
# cv2.imshow("new cell before", for_experiments)
#
# cv2.imshow("mask before", founded_mask)
#
# while len(cs) != 0:
#     for c in cs:
#         result = c.find_nearby_cells(founded_mask, blured_at)
#         cells.extend(result)
#         cs.extend(result)
#         cs.pop(0)
#
# final_image = image.copy()

# for cell in cells:
#     cv2.circle(final_image, (cell.center[0], cell.center[1]), 1, (255, 255, 0), -1)

# cv2.imshow("mask", founded_mask)
#
# #cv2.imshow("final image", final_image)
#
# cv2.imshow("new cell", for_experiments)
# cv2.waitKey(0)

second_experimental = image.copy()

# for c in cells:
#     c.draw_points(second_experimental)
#
# cv2.imshow("new cell after(main picture)", second_experimental)
# cv2.imshow("founded points v 1", points_to_cell_to_image())


# second_experimental = image.copy()
#

cur_to_find = cells.copy()
next_to_find = []

for c in cur_to_find:
    new_cells = c.find_nearby_cells(founded_mask, blured_at)
    if new_cells:
        cells.extend(new_cells)
        next_to_find.extend(new_cells)



while len(next_to_find) != 0:
    cur_to_find = next_to_find.copy()
    next_to_find = []


    for c in cur_to_find:
        new_cells = c.find_nearby_cells(founded_mask, blured_at)
        if new_cells:
            cells.extend(new_cells)
            next_to_find.extend(new_cells)




for c in cells:
    c.draw_points(second_experimental)

# cv2.imshow("new cell after", second_experimental)
# cv2.imshow("founded points v 2", points_to_cell_to_image())
# cv2.imshow("mask", founded_mask)
#
# cv2.waitKey(0)

# unite_points()
#
# cv2.imshow("after founding", for_experiments)
# cv2.imshow("founded", founded_mask)
# cv2.imshow("white points", for_experiments)
#
# cv2.imshow("founded points v 2", points_to_cell_to_image())
#
# cv2.waitKey(0)
#
# founded_mask = np.zeros((800, 800))
#
# NUMBER_OF_CELL = 0
#
# NUMBER_TO_DICT = {}
#
# for c in cells:
#     color_square_and_increase_number(founded_mask, [c.p1, c.p2, c.p3, c.p4])
#
# cv2.imshow("foundedasdasdad", founded_mask)
# cv2.waitKey()

third_experimental = image.copy()

grid = KGrid(cells, NUMBER_TO_DICT, founded_mask)

centers = grid.get_centers()

amount = len(centers)

for i, elem in enumerate(centers):

    for e in elem:
        cv2.circle(third_experimental, (e[0], e[1]), 1, (255, 0, i*(255/amount)), -1)
cv2.imshow("first row", third_experimental)
cv2.waitKey(0)

last_experement = image.copy()

cv2.imshow("result", grid.unwrap(last_experement))
cv2.waitKey(0)

'''