import cv2
import numpy as np
import pickle

from cell import Cell
from Grid import KGrid

LINE_WIDTH = 2

D = 5

SIZEX = 800
SIZEY = 800


def image_read(name, sizex, sizey):
    """Читает изображение заданного размера"""
    image = cv2.imread('Pictures/' + name)
    return image


def get_binary_with_grid(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blured_gray = cv2.blur(gray, (3, 3))

    blured_at = cv2.adaptiveThreshold(blured_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 21, 1)

    something_deleted = use_mask_to_delete(blured_at, iteration=1)

    lineDelete = delete_with_x(blured_at, size=9)

    lineDelete = delete_with_x(lineDelete, size=9)

    all_delete = lineDelete

    repaired = repair_grid(all_delete)

    return repaired


def use_mask_to_delete(binary, iteration=1):
    """Убирает Все тонкие линии шириной меньше 2"""
    sizey, sizex = binary.shape

    left = np.array([[1, 1, -1],
                     [1, 1, -1],
                     [1, 1, -1]])

    right = np.array([[-1, 1, 1],
                     [-1, 1, 1],
                     [-1, 1, 1]])

    up = np.array([[1, 1, 1],
                     [1, 1, 1],
                     [-1, -1, -1]])

    down = np.array([[-1, -1, -1],
                     [1, 1, 1],
                     [1, 1, 1]])

    full = np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1]])


    left_param = cv2.filter2D(binary, cv2.CV_32F, left)

    right_param = cv2.filter2D(binary, cv2.CV_32F, right)

    up_param = cv2.filter2D(binary, cv2.CV_32F, up)

    down_param = cv2.filter2D(binary, cv2.CV_32F, down)

    full = cv2.filter2D(binary, cv2.CV_32F, full)

    result = binary.copy()

    five = 5*255

    all = 8*255

    for y in range(1, binary.shape[0]-1):
        for x in range(1, binary.shape[1]-1):
            if (left_param[y][x] < five and right_param[y][x] < five and \
                    up_param[y][x] < five and down_param[y][x] < five) and full[y][x] < all:
                result[y][x] = 0
    if iteration == 1:
        return result
    else:
        return use_mask_to_delete(binary, iteration=iteration - 1)


def delete_with_x(binary, size=9):
    """Удаляет всё что не прошло тест + образной маской"""
    mask = create_x_mask(size)

    afterMask = cv2.filter2D(binary, cv2.CV_32F, mask)

    for y in range(1, binary.shape[0] - 1):
        for x in range(1, binary.shape[1] - 1):
            if afterMask[y][x] < 255*(size-2):
                binary[y][x] = 0
    return binary


def create_x_mask(size = 5):
    """Возвращает маску в виде + с центром в центре маски"""
    result = np.zeros((size, size))

    for y in range(0, size):
        for x in range(0, size):
            if y == (size-1)/2 or x == (size-1)/2:
                result[y][x] = 1
            else:
                result[y][x] = 0
    return result


def repair_grid(binary):
    """Востанавливает пиксели если на диагонали или вертикали рядом есть несколько пикселей"""
    vMask = np.array([[0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0]])

    gMask = np.array([[0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [1, 1, 0, 1, 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])

    vResult = cv2.filter2D(binary, cv2.CV_32F, vMask)
    gResult = cv2.filter2D(binary, cv2.CV_32F, gMask)

    for y in range(0, binary.shape[0]):
        for x in range(0, binary.shape[1]):
            if vResult[y][x] > 2*255 or gResult[y][x] > 2*255:
                binary[y][x] = 255
    return binary


def get_cells_segments(binary):
    output = cv2.connectedComponentsWithStats(cv2.bitwise_not(binary), 4, cv2.CV_32S)

    num_labels = output[0]

    labels = output[1]

    stats = output[2]

    centroids = output[3]

    masses = []

    only_squares = np.empty((0, 6))

    for i in range(num_labels):
        masses.append(stats[i, cv2.CC_STAT_AREA])
        if stats[i, cv2.CC_STAT_AREA] < 600 and stats[i, cv2.CC_STAT_AREA] > 200:
            only_squares = np.append(only_squares, [np.append(np.int32(stats[i]), i)], axis=0)

    low_labels = labels.copy()

    for y in range(labels.shape[0]):
        for x in range(labels.shape[1]):
            low_labels[y][x] = stats[low_labels[y][x], cv2.CC_STAT_AREA]
            low_labels[y][x] = (low_labels[y][x] < 600 and low_labels[y][x] > 200)


    mean_width = np.mean(only_squares[:, cv2.CC_STAT_WIDTH])
    mean_height = np.mean(only_squares[:, cv2.CC_STAT_HEIGHT])

    only_squares = list(filter(lambda elem: filter_squares(elem, mean_width, mean_height), only_squares))


    result = []

    for elem in only_squares:
        segment = output[1] == elem[5]
        segment = np.multiply(segment, 255)
        segment = np.uint8(segment)
        result.append(segment)

    return result


def filter_squares(elem, mean_width, mean_height):
    """Возвращает true если форма данного элемента не сильно отличается от средней фоормы"""
    width = elem[cv2.CC_STAT_WIDTH]
    height = elem[cv2.CC_STAT_HEIGHT]
    return not(width > mean_width * 1.3 or width < mean_width * 0.7 or
               height > mean_height * 1.3 or height < mean_height * 0.7)



image = image_read("t1.jpg", SIZEX, SIZEY)

print(image.shape)

SIZEY, SIZEX, _ = image.shape

binary = get_binary_with_grid(image)

cv2.imshow("binary", binary)

segments = get_cells_segments(binary)

for_experiments = image.copy()

founded_mask = np.zeros((SIZEY, SIZEX))

print(f'image shape {founded_mask.shape} f_shape {founded_mask.shape}')

cells = []

for segment in segments:

    c = Cell(for_experiments, segment, founded_mask, main=True)
    if c.successful:
        cells.append(c)

for c in cells:
    c.draw_points(for_experiments)

cv2.imshow("draw points", for_experiments)




second_experimental = image.copy()


cur_to_find = cells.copy()
next_to_find = []

for c in cur_to_find:
    new_cells = c.find_nearby_cells(founded_mask, binary)
    if new_cells:
        cells.extend(new_cells)
        next_to_find.extend(new_cells)



while len(next_to_find) != 0:
    cur_to_find = next_to_find.copy()
    next_to_find = []


    for c in cur_to_find:
        new_cells = c.find_nearby_cells(founded_mask, binary)
        if new_cells:
            cells.extend(new_cells)
            next_to_find.extend(new_cells)




for c in cells:
    c.draw_points(second_experimental)

cv2.imshow("new cell after", second_experimental)
cv2.imshow("mask", founded_mask)
cv2.waitKey(0)

third_experimental = image.copy()

grid = KGrid(cells, cells[0].get_number_to_dict(), founded_mask, SIZEX, SIZEY)

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