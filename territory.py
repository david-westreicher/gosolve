import json
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from nn import Classifier
from solver import fill_territory_onlinego

BOARD_SIZE = 19
CROP_SIZE = 32
RECTIFIED_SIZE = 19 * CROP_SIZE
PIECE_SIZE = RECTIFIED_SIZE / BOARD_SIZE
COLORS = {
    'white': 0x00FF00,
    'black': 0x0000FF,
    'unknown': 0xFF0000,
    'X': 0x00FF00,
    'O': 0x0000FF,
}

def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

def get_corners(file_name):
    with open(file_name + '.json', 'r') as f:
        corners = json.load(f)
    return corners

def rectify_image(img, corners):
    start = PIECE_SIZE * 0.5
    end = RECTIFIED_SIZE - PIECE_SIZE * 0.5
    coeffs = find_coeffs(
        [(start, start), (end, start), (end, end), (start, end)],
        corners,
    )
    rectified_image = img.transform((RECTIFIED_SIZE, RECTIFIED_SIZE), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

    H = np.concatenate((coeffs, [1])).reshape((3, 3))

    def apply_h(x, y, h=H):
        p = h.dot(np.asarray([x, y, 1]))
        p /= p[2]
        return p.tolist()[:2]
    '''
    H_inv = np.linalg.inv(H)
    apply_h_inv = lambda x, y: apply_h(x, y, H_inv)
    print(corners)
    print(apply_h(start, start))
    print(apply_h(end, start))
    print(apply_h(end, end))
    print(apply_h(start, end))
    for x, y in corners:
        print(apply_h(x, y, H_inv))
    '''
    return rectified_image, apply_h, H

def draw_lines(img, h=None):
    img = img.copy()
    d = ImageDraw.Draw(img)
    for i in range(BOARD_SIZE):
        # horizontal
        start = (0, (i + 0.5) * PIECE_SIZE)
        end = (RECTIFIED_SIZE, (i + 0.5) * PIECE_SIZE)
        if h:
            start = tuple(h(*start))
            end = tuple(h(*end))
        d.line([start, end], 0xFF0000, 1)
        # vertical
        start = ((i + 0.5) * PIECE_SIZE, 0)
        end = ((i + 0.5) * PIECE_SIZE, RECTIFIED_SIZE)
        if h:
            start = tuple(h(*start))
            end = tuple(h(*end))
        d.line([start, end], 0xFF0000, 1)
    return img

def extract_stones(img):
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            x_0 = x * PIECE_SIZE
            y_0 = y * PIECE_SIZE
            coords = (x_0, y_0, x_0 + PIECE_SIZE, y_0 + PIECE_SIZE)
            crop = img.crop(coords)
            crop = crop.resize((CROP_SIZE, CROP_SIZE), Image.BICUBIC)
            crop = crop.convert('L')
            yield crop, x, y

class ManualPredictor:
    def __init__(self):
        self.img_map = {}
        for folder in ['white', 'black', 'empty']:
            for img_file in os.listdir(f'train/{folder}/'):
                img = Image.open(f'train/{folder}/{img_file}')
                img = np.asarray(img)
                img = tuple(tuple(row) for row in img)
                self.img_map[img] = folder

    def predict(self, img):
        img = np.asarray(img)
        img = tuple(tuple(row) for row in img)
        return self.img_map.get(img, 'unknown')

def draw_stones(img, gf, h=None):
    img = img.copy()
    d = ImageDraw.Draw(img)

    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            stone = gf[y][x]
            if stone == ' ':
                continue
            start = (x * PIECE_SIZE, y * PIECE_SIZE)
            end = ((x + 1) * PIECE_SIZE, (y + 1) * PIECE_SIZE)
            if h:
                start = tuple(h(*start))
                end = tuple(h(*end))
            d.ellipse(start + end, None, COLORS[stone])
    return img

def draw_territory(img, gf, h=None):
    img = img.copy()
    d = ImageDraw.Draw(img, 'RGBA')

    for i in [0, BOARD_SIZE]:
    # for i in range(BOARD_SIZE + 1):
        # horizontal
        start = (0, i * PIECE_SIZE)
        end = (RECTIFIED_SIZE, i * PIECE_SIZE)
        if h:
            start = tuple(h(*start))
            end = tuple(h(*end))
        d.line([start, end], (0, 255, 0, 128), 1)
        # vertical
        start = (i * PIECE_SIZE, 0)
        end = (i * PIECE_SIZE, RECTIFIED_SIZE)
        if h:
            start = tuple(h(*start))
            end = tuple(h(*end))
        d.line([start, end], (0, 255, 0, 128), 1)

    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            stone = gf[y][x]
            if stone == ' ':
                continue
            left = x * PIECE_SIZE
            right = (x + 1) * PIECE_SIZE
            up = y * PIECE_SIZE
            down = (y + 1) * PIECE_SIZE
            lu = (left, up)
            ld = (left, down)
            ru = (right, up)
            rd = (right, down)
            if h:
                lu = tuple(h(*lu))
                ld = tuple(h(*ld))
                ru = tuple(h(*ru))
                rd = tuple(h(*rd))
            col = (0, 0, 0, 128) if stone == 'X' else (255, 255, 255, 128)
            d.polygon([lu, ru, rd, ld], col, None)
    return img

def build_gamefield(predictions):
    gf = [[' '] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    for x, y, pred in predictions:
        if pred is 'empty':
            continue
        if pred == 'white':
            gf[y][x] = 'O'
        if pred == 'black':
            gf[y][x] = 'X'
    return gf

def print_gamefield(gf):
    print('+' + '-' * (BOARD_SIZE * 2 - 1) + '+')
    for row in gf:
        print('|' + ' '.join(row) + '|')
    print('+' + '-' * (BOARD_SIZE * 2 - 1) + '+')

def calc_score(gf):
    score = [0, 0]
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            stone = gf[x][y]
            if stone == ' ':
                continue
            index = 0 if stone == 'X' else 1
            score[index] += 1
    return score

def draw_score(img, score):
    text_size = int(max(img.size) / 40.0)
    font = ImageFont.truetype("Verdana.ttf", text_size)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, text_size * 2, text_size * 2], 0x666666)
    d.text((0, 0), str(score[0]).zfill(3), 0x000000, font)
    d.text((0, text_size), str(score[1]).zfill(3), 0xFFFFFF, font)
    return img

def calculate_territory(img_path, corners, predictor=Classifier()):
    print(img_path, corners)
    img = Image.open(img_path)
    rect_img, homography, H = rectify_image(img, corners)
    stones = extract_stones(rect_img)
    predictions = [(x, y, predictor.predict(stone)) for stone, x, y in stones]
    gf = build_gamefield(predictions)
    area = fill_territory(gf)
    score = calc_score(area)
    return gf, area, H.tolist(), score


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('file')
    parser.add_argument('--extract_stones', action='store_true')
    parser.add_argument('--corners')
    args = parser.parse_args()
    print(args)

    input_file = args.file
    img = Image.open(input_file)
    
    if args.corners:
        corners = args.corners.split(',')
        corners = [float(e) for e in corners]
        corners = list(zip(corners[::2], corners[1::2]))
    else:
        corners = get_corners(input_file)
    rect_img, homography, H = rectify_image(img, corners)
    debug_img = draw_lines(rect_img)
    # debug_img.save(input_file + '_debug2.png')

    stones = extract_stones(rect_img)
    if args.extract_stones:
        label_map = {}
        for folder in ['white', 'black', 'empty', 'unknown']:
            for img_file in os.listdir(f'train/{folder}/'):
                label_map[img_file] = folder

        img_name = os.path.basename(input_file)
        for stone, x, y in stones:
            stone_name = f'{img_name}_{x}_{y}.png'
            label = label_map.get(stone_name, 'unknown')
            stone.save(f'train/{label}/{stone_name}')

    predictor = Classifier()
    predictions = [(x, y, predictor.predict(stone)) for stone, x, y in stones]
    gf = build_gamefield(predictions)
    debug_img = draw_stones(debug_img, gf)
    # debug_img.save(input_file + '_debug3.png')

    debug_img = draw_territory(img, gf, homography)
    score = calc_score(gf)
    debug_img = draw_score(debug_img, score)
    debug_img.save(input_file + '_debug1.png')

    print_gamefield(gf)
    gf = fill_territory_onlinego(gf)
    print_gamefield(gf)
    debug_img = draw_stones(debug_img, gf)
    # debug_img.save(input_file + '_debug4.png')

    debug_img = draw_territory(img, gf, homography)
    score = calc_score(gf)
    debug_img = draw_score(debug_img, score)
    debug_img.save(input_file + '_debug2.png')
