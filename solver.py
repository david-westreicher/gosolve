import subprocess
import tempfile
from copy import deepcopy

def free_connected_components(gf):
    visited = set()

    def visit(x, y, comp):
        if (x, y) in visited:
            return
        visited.add((x, y))
        comp.add((x, y))
        for nx, ny in [[1, 0], [0, 1], [-1, 0], [0, -1]]:
            ox = x + nx
            oy = y + ny
            if not(0 <= ox < len(gf) and 0 <= oy < len(gf)):
                continue
            neigh = gf[ox][oy]
            if neigh != ' ':
                continue
            visit(ox, oy, comp)

    comps = []
    for x in range(len(gf)):
        for y in range(len(gf)):
            if (x, y) in visited or gf[x][y] != ' ':
                continue
            comp = set()
            visit(x, y, comp)
            comps.append(comp)

    '''
    for c in comps:
        print(c)
    '''
    return comps

def neighbors(gf, comp):
    neighs = set()
    for x, y in comp:
        for nx, ny in [[1, 0], [0, 1], [-1, 0], [0, -1]]:
            ox = x + nx
            oy = y + ny
            if not(0 <= ox < len(gf) and 0 <= oy < len(gf)):
                continue
            neigh = gf[ox][oy]
            if neigh == ' ':
                continue
            neighs.add(neigh)
    return neighs

def fill_territory(gf):
    gf = deepcopy(gf)
    free_comps = free_connected_components(gf)
    for num, comp in enumerate(free_comps):
        neighs = neighbors(gf, comp)
        if len(neighs) != 1:
            continue
        color, = neighs
        for x, y in comp:
            gf[x][y] = color
    return gf

def fill_territory_onlinego(gf):
    val_map = {
        'X': ' 1',
        'O': '-1',
        ' ': ' 0',
    }
    online_gf_map = {
        '_': 'O',
        'o': 'X',
        ' ': ' ',
    }
    with tempfile.NamedTemporaryFile(mode='w') as tmp_file:
        tmp_file.write('# 1=black -1=white 0=open\n')
        tmp_file.write(f'height {len(gf)}\n')
        tmp_file.write(f'width {len(gf)}\n')
        tmp_file.write(f'player_to_move {-1}\n')
        for row in gf:
            row = [val_map[e] for e in row]
            row = ' '.join(row)
            tmp_file.write(f'{row}\n')
        tmp_file.flush()
        print(tmp_file.name)
        cmd = ['score-estimator/estimator', tmp_file.name, '10000']
        print(' '.join(cmd))
        output = subprocess.check_output(cmd).decode()

    output_gf = []
    for i, line in enumerate(output.split('\n')):
        row = line[::2]
        row = [online_gf_map[e] for e in row]
        output_gf.append(row)
        if len(output_gf) == len(gf):
            break
    return output_gf
