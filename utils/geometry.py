from itertools import permutations

def unique_rotations_3d(box):
    w, h, d = box
    perms = set(permutations((w, h, d), 3))
    return sorted(list(perms))

def aabb_overlap(a, b):
    ax, ay, az, aw, ah, ad = a
    bx, by, bz, bw, bh, bd = b
    return not (ax + aw <= bx or bx + bw <= ax or
                ay + ah <= by or by + bh <= ay or
                az + ad <= bz or bz + bd <= az)

def supported_on_floor_or_stack(a, placed):
    x, y, z, w, h, d = a
    if z == 0:
        return True
    # Require full support directly beneath footprint at z-1 layer
    # We check if the union of top faces of placed boxes at height z' matches footprint.
    # Simplified: ensure at least one box provides support area under (x..x+w, y..y+h) at z' = z-1 or touching top.
    # For LfD, a simple heuristic: require there exists at least one box below that overlaps >= 80% of footprint.
    footprint = w * h
    if footprint == 0:
        return False
    covered = 0
    for b in placed:
        bx, by, bz = b["pos"]
        bw, bh, bd = b["size"]
        top_z = bz + bd
        if top_z != z:
            continue
        ox = max(0, min(x + w, bx + bw) - max(x, bx))
        oy = max(0, min(y + h, by + bh) - max(y, by))
        overlap = max(0, ox) * max(0, oy)
        covered += overlap
    return covered >= 0.8 * footprint

def enumerate_floor_anchors_for_rotation(bin_size, box_size):
    W, H, D = bin_size
    w, h, d = box_size
    anchors = []
    for x in range(0, W - w + 1):
        for y in range(0, H - h + 1):
            anchors.append((x, y, 0))
    return anchors

def enumerate_all_anchors_for_rotation(bin_size, placed, box_size):
    W, H, D = bin_size
    w, h, d = box_size
    anchors = []
    # Floor anchors
    for x in range(0, W - w + 1):
        for y in range(0, H - h + 1):
            anchors.append((x, y, 0))
    # Stack anchors: try top faces of placed boxes
    for b in placed:
        bx, by, bz = b["pos"]
        bw, bh, bd = b["size"]
        top_z = bz + bd
        if top_z + d > D:
            continue
        # slide over top face
        for x in range(bx, bx + bw - w + 1):
            if x < 0 or x + w > W:
                continue
            for y in range(by, by + bh - h + 1):
                if y < 0 or y + h > H:
                    continue
                anchors.append((x, y, top_z))
    # Deduplicate
    anchors = sorted(list(set(anchors)))
    return anchors
