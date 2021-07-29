import math
import numpy as np




def rotate_matrix(ry):
    '''yam _rotate_matrix
    '''
    R = np.array([[math.cos(ry), 0, math.sin(ry)], 
                  [0, 1, 0], 
                  [-math.sin(ry), 0, math.cos(ry)]])
    return R


def _corners(loc, dim, ry, K, origin):
    '''
    y z x(h w l)(kitti label file) <-> x y z(l h w)(camera)
    '''
    l, h, w = dim
    R = rotate_matrix(ry)

    _x = np.array([l, l, 0, 0, l, l, 0, 0]) - l * origin[0]
    _y = np.array([h, h, h, h, 0, 0, 0, 0]) - h * origin[1]
    _z = np.array([w, 0, 0, w, w, 0, 0, w]) - w * origin[2]
    _corners_3d = np.vstack([_x, _y, _z])
    _corners_3d = R @ _corners_3d + np.array(loc).reshape(3, 1) # 3 x 8

    return _corners_3d.T 
    
    
def build_corners(center, depth, size, rs, K, origin=(0.5, 0.5, 0.5)):
    '''Corners
    args:
        n x 2 [x, y]
        n x 1
        n x 3 [l, h, w]
        n x 1
        n x 4 x 4
    return 
        n x 8 x 3
    '''
    if center.shape[-1] == 2:
        center = np.concatenate((center, np.ones((center.shape[0], 1))), axis=-1)
    center *= depth.reshape(-1, 1)
    
    K_inv = np.linalg.pinv(K) # n x 4 x 4
    
    center_3d = K_inv[:, :3, :3] @ center[:, :, None] # n x 3 x 1
    
    corner_3d = []
    for loc, dim, ry, k in zip(center_3d, size, rs, K):
        corner_3d.append(_corners(loc, dim, ry, k, origin))
    
    corner_3d = np.array(corner_3d) # n, 8, 3
    
    return corner_3d



def point3d_to_image(points, K):
    '''
    args:
        n x m x 3
        n x 4 x 4
    return:
        n x m x 2
    '''
    _points = K[:, :3, :3] @ points.transpose(0, 2, 1)
    _points = _points.transpose(0, 2, 1)
    _points[:, :, [0, 1]] /= _points[:, :, [2]]
    
    return _points[:, :, [0, 1]]


