import skimage.measure
import trimesh
import open3d as o3d
import numpy as np

def marching_cubes(occupancy, level=0.5):
    try:
        vertices, faces, vertex_normals, _ = skimage.measure.marching_cubes_lewiner(    #marching_cubes(
            occupancy, level=level, gradient_direction='ascent')
    except (RuntimeError, ValueError):
       return None
    # vertices, faces, vertex_normals, _ = skimage.measure.marching_cubes_lewiner(  # marching_cubes(
    #     occupancy, level=level, gradient_direction='ascent')
    # todo try pytorch3d marching cube to speed up?
    # https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/marching_cubes.py
    # vertices, faces = mcubes.marching_cubes(occupancy, level)

    # vertices, faces = mc.mcubes_cuda(1. - occupancy, 0.5)
    # vertices = vertices.cpu().numpy()
    # faces = faces.cpu().numpy()

    dim = occupancy.shape[0]
    vertices = vertices / (dim - 1)
    mesh = trimesh.Trimesh(vertices=vertices,
                           vertex_normals=vertex_normals,
                           faces=faces)

    return mesh

def trimesh_to_open3d(src):
    dst = o3d.geometry.TriangleMesh()
    dst.vertices = o3d.utility.Vector3dVector(src.vertices)
    dst.triangles = o3d.utility.Vector3iVector(src.faces)
    vertex_colors = src.visual.vertex_colors[:, :3].astype(np.float) / 255.0
    dst.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    dst.compute_vertex_normals()

    return dst