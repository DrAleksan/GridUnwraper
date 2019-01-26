import numpy as np
import cv2
import cmath
import matplotlib.pyplot as plt
import pickle



NUMBER_OF_CELL = 1  # Номер текущей клетки
NUMBER_TO_DICT = {}  # Map, отображающий номер клетки в клетку

founded_mask = None



POINT_TO_CELL = {}  # Map, отображающий точку в пару (клетка, номер точки)

LINE_WIDTH = 2

D = 3

SIZEX = 800
SIZEY = 800

thresholded = None


class Cell:
    def __init__(self, mask, input_founded_mask, build=True, main=False):
        global SIZEX
        global SIZEY
        global founded_mask
        founded_mask = input_founded_mask
        SIZEY, SIZEX = founded_mask.shape
        self.main = main
        self.mask = mask
        self.lifetime = 3

        self.p1, self.p2, self.p3, self.p4, self.successful = 0, 0, 0, 0, True
        self.d1, self.d2, self.d3, self.d4 = 0, 0, 0, 0
        self.center = [0, 0]
        self.len_h = 0
        self.len_w = 0
        self.number = -1

        self.top_neighborhood = None
        self.bottom_neighborhood = None
        self.right_neighborhood = None
        self.left_neighborhood = None

        if build:
            self.calc_all()

            if not self.successful:
                return

            NUMBER_TO_DICT[NUMBER_OF_CELL] = self
            self.number = NUMBER_OF_CELL
            color_square_and_increase_number(founded_mask, [self.p1, self.p2, self.p3, self.p4])

    def calc_all(self):
        self.p1, self.p2, self.p3, self.p4, self.successful = self.find_points(self.mask)

        if not self.successful:
            return
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


        d1 = build_vector(p1, p2)
        d11 = build_vector(p4, p3)

        d2 = build_vector(p2, p1)

        d3 = build_vector(p4, p1)
        d31 = build_vector(p3, p2)

        d4 = build_vector(p1, p4)

        dist1 = np.linalg.norm(d1)
        dist11 = np.linalg.norm(d11)

        dist3 = np.linalg.norm(d3)
        dist31 = np.linalg.norm(d31)

        if dist1 != dist11 or dist3 != dist31:
            self.successful = False
            return 0, 0, 0, 0

        if np.dot(d1, d11)/(dist1*dist11) < 0.99 or np.dot(d3, d31)/(dist3*dist31) < 0.99:
            self.successful = False
            return 0, 0, 0, 0


        return d1, d2, d3, d4


    def adjust_point(self, x, y, number):
        cell, n = POINT_TO_CELL[(y, x)][0]
        point = cell.get_point(n)

        new_x = int((point[0] + self.get_point(number)[0]) / 2)
        new_y = int((point[1] + self.get_point(number)[1]) / 2)

        POINT_TO_CELL[(y, x)].append((self, number))


        value = POINT_TO_CELL[(y, x)]

        for elem in POINT_TO_CELL[(y, x)]:
            points_before = elem[0].get_points()
            elem[0].set_point(new_x, new_y, elem[1])
            points_after = elem[0].get_points()
            # recolor_square(points_before, points_after, elem[0])
        clean_POINT_TO_CELL((new_x, new_y))

        for j in range(new_y - D, new_y + D):
            for i in range(new_x - D, new_x + D):
                POINT_TO_CELL[(j, i)] = value


    def draw_cell(self):
        all_p_x = [self.p1[0], self.p2[0], self.p3[0], self.p4[0]]
        all_p_y = [self.p1[1], self.p2[1], self.p3[1], self.p4[1]]
        origin = [np.mean(all_p_x)], [np.mean(all_p_y)]

        print(self.d1)

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

    def find_nearby_cells(self, founded_mask, thresholded):
        if self.lifetime == 0:
            return None

        dir1 = (self.p1[0] - self.p2[0], self.p1[1] - self.p2[1])
        dir2 = (self.p2[0] - self.p1[0], self.p2[1] - self.p1[1])

        dir3 = (self.p4[0] - self.p1[0], self.p4[1] - self.p1[1])
        dir4 = (self.p1[0] - self.p4[0], self.p1[1] - self.p4[1])

        c1 = (int(self.center[0] + dir1[0]), int(self.center[1] + dir1[1]))
        c2 = (int(self.center[0] + dir2[0]), int(self.center[1] + dir2[1]))
        c3 = (int(self.center[0] + dir3[0]), int(self.center[1] + dir3[1]))
        c4 = (int(self.center[0] + dir4[0]), int(self.center[1] + dir4[1]))


        result = []

        if 0 < c1[0] < SIZEX and 0 < c1[1] < SIZEY and founded_mask[c1[1]][c1[0]] == 0:
            new_upper_cell = Cell(self.mask,
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
            new_upper_cell.lifetime = self.lifetime-1

            new_upper_cell.bind_points(new_upper_cell.p1, new_upper_cell.p2, new_upper_cell.p3, new_upper_cell.p4)

            NUMBER_TO_DICT[NUMBER_OF_CELL] = new_upper_cell
            new_upper_cell.number = NUMBER_OF_CELL
            color_square_and_increase_number(founded_mask, [new_upper_cell.p1, new_upper_cell.p2,
                                                            new_upper_cell.p3, new_upper_cell.p4])

            if(new_upper_cell.check()):
                result.append(new_upper_cell)
                self.top_neighborhood = new_upper_cell
            else:
                print("bad cell")


        if 0 < c2[0] < SIZEX and 0 < c2[1] < SIZEY and founded_mask[c2[1]][c2[0]] == 0:
            # cv2.circle(self.image, (c1[0], c1[1]), 3, (255, 255, 255), -1)
            new_bottom_cell = Cell(self.mask,
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
            new_bottom_cell.lifetime = self.lifetime - 1

            new_bottom_cell.bind_points(new_bottom_cell.p1, new_bottom_cell.p2, new_bottom_cell.p3, new_bottom_cell.p4)

            NUMBER_TO_DICT[NUMBER_OF_CELL] = new_bottom_cell
            new_bottom_cell.number = NUMBER_OF_CELL
            color_square_and_increase_number(founded_mask, [new_bottom_cell.p1, new_bottom_cell.p2,
                                                            new_bottom_cell.p3, new_bottom_cell.p4])

            if(new_bottom_cell.check()):
                result.append(new_bottom_cell)
                self.bottom_neighborhood = new_bottom_cell
            else:
                print("bad cell")


        if 0 < c3[0] < SIZEX and 0 < c3[1] < SIZEY and founded_mask[c3[1]][c3[0]] == 0:
            # cv2.circle(self.image, (c1[0], c1[1]), 3, (255, 255, 255), -1)
            new_right_cell = Cell(self.mask,
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
            new_right_cell.lifetime = self.lifetime - 1

            # print(f'{} {} {} {}')

            new_right_cell.bind_points(new_right_cell.p1, new_right_cell.p2, new_right_cell.p3, new_right_cell.p4)

            NUMBER_TO_DICT[NUMBER_OF_CELL] = new_right_cell
            new_right_cell.number = NUMBER_OF_CELL
            color_square_and_increase_number(founded_mask, [new_right_cell.p1, new_right_cell.p2,
                                                            new_right_cell.p3, new_right_cell.p4])
            if(new_right_cell.check()):
                result.append(new_right_cell)
                self.right_neighborhood = new_right_cell
            else:
                print("bad cell")

        if 0 < c4[0] < SIZEX and 0 < c4[1] < SIZEY and founded_mask[c4[1]][c4[0]] == 0:
            # cv2.circle(self.image, (c1[0], c1[1]), 3, (255, 255, 255), -1)
            new_left_cell = Cell(self.mask,
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
            new_left_cell.lifetime = self.lifetime - 1

            # print(f'{} {} {} {}')

            new_left_cell.bind_points(new_left_cell.p1, new_left_cell.p2, new_left_cell.p3, new_left_cell.p4)

            NUMBER_TO_DICT[NUMBER_OF_CELL] = new_left_cell
            new_left_cell.number = NUMBER_OF_CELL
            color_square_and_increase_number(founded_mask, [new_left_cell.p1, new_left_cell.p2,
                                                            new_left_cell.p3, new_left_cell.p4])
            if(new_left_cell.check()):
                result.append(new_left_cell)
                self.left_neighborhood = new_left_cell
            else:
                print("bad cell")

        return result


    def draw_points(self, image):
        cv2.circle(image, (int(self.p1[0]), int(self.p1[1])), 1, (255, 255, 0), -1)
        cv2.circle(image, (int(self.p2[0]), int(self.p2[1])), 1, (0, 255, 255), -1)
        cv2.circle(image, (int(self.p3[0]), int(self.p3[1])), 1, (0, 255, 0), -1)
        cv2.circle(image, (int(self.p4[0]), int(self.p4[1])), 1, (0, 0, 255), -1)
        cv2.circle(image, (self.center[0], self.center[1]), 1, (255, 0, 0), -1)

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

    def get_points(self):
        return (self.p1, self.p2, self.p3, self.p4)

    def get_bottom_neighborhood(self):
        if self.bottom_neighborhood != None:
            return self.bottom_neighborhood
        else:
            pos = (self.center[0] + self.d2[0], self.center[1] + self.d2[1])
            if 0 < pos[1] < SIZEY and 0 < pos[0] < SIZEX and founded_mask[pos[1]][pos[0]] != 0:
                self.bottom_neighborhood = NUMBER_TO_DICT[founded_mask[pos[1]][pos[0]]]
                return NUMBER_TO_DICT[founded_mask[pos[1]][pos[0]]]
    def get_top_neighborhood(self):
        if self.top_neighborhood != None:
            return self.top_neighborhood
        else:
            pos = (self.center[0] + self.d1[0], self.center[1] + self.d1[1])
            if 0 < pos[1] < SIZEY and 0 < pos[0] < SIZEX and founded_mask[pos[1]][pos[0]] != 0:
                self.top_neighborhood = NUMBER_TO_DICT[founded_mask[pos[1]][pos[0]]]
                return NUMBER_TO_DICT[founded_mask[pos[1]][pos[0]]]
    def get_right_neighborhood(self):
        if self.right_neighborhood != None:
            return self.right_neighborhood
        else:
            pos = (self.center[0] + self.d3[0], self.center[1] + self.d3[1])
            if 0 < pos[1] < SIZEY and 0 < pos[0] < SIZEX and founded_mask[pos[1]][pos[0]] != 0:
                self.right_neighborhood = NUMBER_TO_DICT[founded_mask[pos[1]][pos[0]]]
                return NUMBER_TO_DICT[founded_mask[pos[1]][pos[0]]]
    def get_left_neighborhood(self):
        if self.left_neighborhood != None:
            return self.left_neighborhood
        else:
            pos = (self.center[0] + self.d4[0], self.center[1] + self.d4[1])
            if 0 < pos[1] < SIZEY and 0 < pos[0] < SIZEX and founded_mask[pos[1]][pos[0]] != 0:
                self.left_neighborhood = NUMBER_TO_DICT[founded_mask[pos[1]][pos[0]]]
                return NUMBER_TO_DICT[founded_mask[pos[1]][pos[0]]]

    def check(self):
        d1 = build_vector(self.p1, self.p2)
        d11 = build_vector(self.p4, self.p3)

        d3 = build_vector(self.p4, self.p1)
        d31 = build_vector(self.p3, self.p2)

        dist1 = np.linalg.norm(d1)
        dist11 = np.linalg.norm(d11)

        dist3 = np.linalg.norm(d3)
        dist31 = np.linalg.norm(d31)

        if abs(dist1 - dist11) > 1 or abs(dist3 - dist31) > 1:
            return False

        print(f'c1 {np.dot(d1, d11)/(dist1*dist11)} c2 {np.dot(d3, d31)/(dist3*dist31)}')

        if np.dot(d1, d11)/(dist1*dist11) < 0.99 or np.dot(d3, d31)/(dist3*dist31) < 0.99:
            return False
        return True

    def get_number_to_dict(self):
        return NUMBER_TO_DICT

    # def count_inside_cell(self, number):
    #     result = 0
    #
    #     for j in range(SIZEY):
    #         for i in range(SIZEX):
    #             result +=


def build_vector(p1, p2):
    return (p1[0] - p2[0], p1[1] - p2[1])

def asign_area(cell, x, y, number):
    for j in range(y-D, y+D):
        for i in range(x-D, x+D):
            POINT_TO_CELL[(j, i)] = [(cell, number)]


def recolor_square(points_before, points_after, number):
    global founded_mask
    points_before = np.array(points_before, 'int32')

    cv2.fillConvexPoly(founded_mask, points_before, 0)

    points_after = np.array(points_after, 'int32')

    cv2.fillConvexPoly(founded_mask, points_after, number)


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


def clean_POINT_TO_CELL(point):
    for j in range(point[1] - 2*D, point[1] + 2*D):
        for i in range(point[0] - 2*D, point[0] + 2*D):
            if (j, i) in POINT_TO_CELL:
                del POINT_TO_CELL[(j, i)]


#
# with open('saved_variables_t1', 'rb') as f:
#     image, blured_at, all_delete, repaired, output, only_squares, left_top_coords, mean_width, mean_height, final_mask = pickle.load(
#         f)
#
# # cv2.waitKey(0)
#
# founded_mask = np.zeros((800, 800))
#
# cells = []
#
#
# for_experiments = image.copy()
#
# for elem in only_squares:
#
#     segment = output[1] == elem[5]
#     segment = np.multiply(segment, 255)
#     segment = np.uint8(segment)
#
#     c = Cell(segment, founded_mask, build=True, main=True)
#     if c.successful:
#         cells.append(c)
#
# for c in cells:
#     c.draw_points(for_experiments)
#
# cv2.imshow("draw points", for_experiments)
# cv2.waitKey(0)