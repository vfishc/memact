"""Physical side lengths from the exact transformed bounding box used in rendering."""

import hashlib
import json
import math


def mesh_size_truth(geometry):
    corners = geometry['world_obb_corners']
    if len(corners) != 8 or any(len(point) != 3 or not all(math.isfinite(value) for value in point) for point in corners):
        raise ValueError('Invalid transformed mesh bounding box')
    origin = corners[0]
    vectors = [[corners[index][axis] - origin[axis] for axis in range(3)] for index in [1, 3, 4]]
    lengths = [math.sqrt(sum(value * value for value in vector)) for vector in vectors]
    if min(lengths) <= 0:
        raise ValueError('Degenerate transformed mesh bounding box')
    for first in range(3):
        for second in range(first + 1, 3):
            cosine = sum(vectors[first][axis] * vectors[second][axis] for axis in range(3)) / (lengths[first] * lengths[second])
            if abs(cosine) > 1e-4:
                raise ValueError('Sheared mesh box requires explicit size semantics')
    return {'source': 'rendered_mesh_transformed_local_bbox_edges_v1',
            'side_lengths_m': lengths, 'longest_side_m': max(lengths),
            'geometry_sha256': hashlib.sha256(json.dumps(geometry, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}
