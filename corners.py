import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
from vis import Vis
from scipy.ndimage.filters import convolve1d, sobel
from scipy.signal import convolve2d

SCANLINE_OFFSET = 20
IMG_SIZE = 640
LINES_TO_EXTRACT = 0

def calc_gradient(img, pos):
    if not (img.shape[0] - 2 > pos[0] >= 2 and img.shape[1] - 2 > pos[1] >= 2):
        # TODO return None
        return None
    sub_img = img[pos[0] - 1: pos[0] + 2, pos[1] - 1: pos[1] + 2]
    sx = sobel(sub_img, axis=0)[1][1]
    sy = sobel(sub_img, axis=1)[1][1]
    if sy == 0:
        grad = np.pi * 0.5
    else:
        grad = np.arctan2(sx, sy)
    grad_vec = rot_vec(np.asarray([0, 1]), grad)
    return grad_vec
    '''
    g_x = np.dot(img[pos[0], pos[1] - 2: pos[1] + 3], [1, 3, -8, 3, 1])
    g_y = np.dot(img[pos[0] - 2: pos[0] + 3, pos[1]], [1, 3, -8, 3, 1])
    if g_x == 0:
        grad = np.pi * 0.5
    else:
        # TODO diagonal filter for low magnitude: maybe rot90?
        grad = np.arctan2(-g_y, g_x)
    grad_vec = rot_vec(np.asarray([0, 1]), grad)
    if grad_vec[0] <= 0:
        grad_vec *= -1.0
    return grad_vec
    '''

def rot_vec(vec, rad):
    c, s = np.cos(rad), np.sin(rad)
    R = np.array(((c, -s), (s, c)))
    return np.dot(R, vec)

def distance(p1, p2, p):
    x1, y1 = p1
    x2, y2 = p2
    x0, y0 = p
    nom = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denom = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    return nom / denom

def normalize(v):
    return v / np.linalg.norm(v)


vis = Vis(lambda x: x)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('file')
    args = parser.parse_args()
    print(args)

    input_file = args.file
    img = Image.open(input_file)
    img = img.resize((IMG_SIZE, round(float(IMG_SIZE) * img.size[1] / img.size[0])), Image.BICUBIC)
    img = img.convert('RGB')

    d = ImageDraw.Draw(img)
    np_img = np.asarray(img.convert('L'), dtype=float)
    np_img[np_img < 100] = 0
    np_img[np_img > 200] = 200
    print(np_img.shape)
    points = []
    for orientation in ['hor', 'ver']:
        if orientation == 'hor':
            rows = np_img[0::SCANLINE_OFFSET]
        else:
            rows = np_img.T[0::SCANLINE_OFFSET]
        rows = convolve1d(rows, [-3, -5, 0, 5, 3])
        rows[abs(rows) < 400] = 0
        if orientation == 'ver':
            rows = rows.T

        for x in range(rows.shape[0]):
            for y in range(rows.shape[1]):
                if rows[x][y] == 0:
                    continue
                if orientation == 'hor':
                    pos = (y, x * SCANLINE_OFFSET)
                else:
                    pos = (y * SCANLINE_OFFSET, x)
                pos = np.asarray(pos)
                grad = calc_gradient(np_img, [pos[1], pos[0]])
                if grad is None:
                    continue
                start = (pos[0] - 1, pos[1] - 1)
                end = (pos[0] + 1, pos[1] + 1)
                new_pos = pos + grad * 10.0
                points.append((pos, grad))
                # TODO filter by gradient magnitude
                d.ellipse(start + end, int((0xFF0000 if grad is not None else 0x000000) * 0.5), None)
                d.line([tuple(pos), tuple(new_pos)], 0xFFFFFF)

    for _ in range(LINES_TO_EXTRACT):
        max_support = set()
        support_line = None
        for _ in range(1000):
            (p1, n1), (p2, n2) = random.sample(points, k=2)
            if tuple(p1) == tuple(p2):
                continue
            if abs(np.dot(n1, n2)) < 0.95:
                continue
            
            if abs(np.dot(normalize(p1 - p2), n2)) < 0.95:
                continue
            support = set()
            for p3, n3 in points:
                if distance(p1, p2, p3) > 1.5:
                    continue
                if abs(np.dot(n1, n3)) < 0.95:
                    continue
                support.add(tuple(p3))
            if len(support) > len(max_support):
                max_support = support
                support_line = (p1, p2)
        if len(max_support) < 4:
            break
        points = [(p, n) for p, n in points if tuple(p) not in max_support]
        p1, p2 = support_line
        print(max_support)
        for pos in max_support:
            start = (pos[0] - 1, pos[1] - 1)
            end = (pos[0] + 1, pos[1] + 1)
            d.ellipse(start + end, 0xFF0000, None)
        # d.line([tuple(p1 - (p1 - p2) * 1000), tuple(p1 + (p1 - p2) * 1000)], 0xFFFFFF)
        # TODO regress nice line through support
        d.line([tuple(p1 + (p1 - p2) * 100), tuple(p1 + (p1 - p2) * -100)], 0xFFFFFF)

    # print(np.asarray(img).shape)
    img.save('tmp/debug.png')
    vis.showimg(np.asarray(img.convert('L')))
    vis.showimg(np_img)
