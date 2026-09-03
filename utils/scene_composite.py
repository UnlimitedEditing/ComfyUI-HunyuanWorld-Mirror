"""
Composite scene assembly (textured per-view mesh + camera-frustum markers, in
one GLB), ported from Tencent's official HunyuanWorld-Mirror demo
(src/utils/visual_util.py -- create_image_mesh, integrate_camera_into_scene,
apply_transformation_to_points, generate_camera_mesh_faces,
convert_predictions_to_glb_scene -- fetched and ported directly).

Deliberately built from the raw per-pixel `points3d`/`depth`/image data
(HWMInference's own output), NOT from the exported Gaussians -- the model's
internal prune_gs voxel-merge reorders/dedupes Gaussians before this node pack
ever sees them, so there's no per-pixel correspondence left to build a
textured mesh or apply masks against on the splat path (same reasoning
already applied to skip edge-masking on Save3DGaussians).
"""

from typing import Tuple

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def _convert_quads_to_triangles(quad_faces: np.ndarray) -> np.ndarray:
    if quad_faces.shape[-1] == 3:
        return quad_faces
    num_vertices_per_face = quad_faces.shape[-1]
    triangle_indices = np.stack([
        np.zeros(num_vertices_per_face - 2, dtype=int),
        np.arange(1, num_vertices_per_face - 1, dtype=int),
        np.arange(2, num_vertices_per_face, dtype=int),
    ], axis=1)
    return quad_faces[:, triangle_indices].reshape((-1, 3))


def create_image_mesh(
    *image_data: np.ndarray,
    mask: np.ndarray = None,
    triangulate: bool = False,
    return_vertex_indices: bool = False,
) -> Tuple[np.ndarray, ...]:
    """Turns pixel coordinates into vertices and grid-neighbor connections into faces --
    i.e. a depth-map/point-map + image becomes a real textured mesh 'card', not just points."""
    assert (len(image_data) > 0) or (mask is not None), "Need at least one image or mask"

    if mask is None:
        height, width = image_data[0].shape[:2]
    else:
        height, width = mask.shape

    for img in image_data:
        assert img.shape[:2] == (height, width), "All images must have same height and width"

    base_quad = np.stack([
        np.arange(0, width - 1, dtype=np.int32),
        np.arange(width, 2 * width - 1, dtype=np.int32),
        np.arange(1 + width, 2 * width, dtype=np.int32),
        np.arange(1, width, dtype=np.int32),
    ], axis=1)

    row_offsets = np.arange(0, (height - 1) * width, width, dtype=np.int32)
    faces = (row_offsets[:, None, None] + base_quad[None, :, :]).reshape((-1, 4))

    if mask is None:
        if triangulate:
            faces = _convert_quads_to_triangles(faces)
        output = [faces]
        for img in image_data:
            output.append(img.reshape(-1, *img.shape[2:]))
        if return_vertex_indices:
            output.append(np.arange(height * width, dtype=np.int32))
        return tuple(output)
    else:
        valid_quads = (
            mask[:-1, :-1] & mask[1:, :-1] &
            mask[1:, 1:] & mask[:-1, 1:]
        ).ravel()
        faces = faces[valid_quads]
        if triangulate:
            faces = _convert_quads_to_triangles(faces)

        num_face_vertices = faces.shape[-1]
        unique_vertices, remapped_indices = np.unique(faces, return_inverse=True)
        faces = remapped_indices.astype(np.int32).reshape(-1, num_face_vertices)

        output = [faces]
        for img in image_data:
            flattened_img = img.reshape(-1, *img.shape[2:])
            output.append(flattened_img[unique_vertices])
        if return_vertex_indices:
            output.append(unique_vertices)
        return tuple(output)


def apply_transformation_to_points(transform_matrix: np.ndarray, point_array: np.ndarray, output_dim: int = None) -> np.ndarray:
    point_array = np.asarray(point_array)
    original_shape = point_array.shape[:-1]
    target_dim = output_dim or point_array.shape[-1]
    transposed_transform = transform_matrix.swapaxes(-1, -2)
    transformed_points = (
        point_array @ transposed_transform[..., :-1, :] +
        transposed_transform[..., -1:, :]
    )
    return transformed_points[..., :target_dim].reshape(*original_shape, target_dim)


def generate_camera_mesh_faces(base_cone_mesh: trimesh.Trimesh) -> np.ndarray:
    face_indices = []
    vertex_count_per_cone = len(base_cone_mesh.vertices)

    for triangle_face in base_cone_mesh.faces:
        if 0 in triangle_face:
            continue
        vertex_a, vertex_b, vertex_c = triangle_face
        vertex_a_layer2, vertex_b_layer2, vertex_c_layer2 = triangle_face + vertex_count_per_cone
        vertex_a_layer3, vertex_b_layer3, vertex_c_layer3 = triangle_face + 2 * vertex_count_per_cone

        connecting_faces = [
            (vertex_a, vertex_b, vertex_b_layer2),
            (vertex_a, vertex_a_layer2, vertex_c),
            (vertex_c_layer2, vertex_b, vertex_c),
            (vertex_a, vertex_b, vertex_b_layer3),
            (vertex_a, vertex_a_layer3, vertex_c),
            (vertex_c_layer3, vertex_b, vertex_c),
        ]
        face_indices.extend(connecting_faces)

    reversed_faces = [(c, b, a) for a, b, c in face_indices]
    face_indices.extend(reversed_faces)
    return np.array(face_indices)


def integrate_camera_into_scene(scene: trimesh.Scene, camera_transform: np.ndarray, camera_color: tuple, scale_factor: float):
    camera_base_width = scale_factor * 0.05
    camera_cone_height = scale_factor * 0.1

    base_cone = trimesh.creation.cone(camera_base_width, camera_cone_height, sections=4)

    z_rotation_matrix = np.eye(4)
    z_rotation_matrix[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    z_rotation_matrix[2, 3] = -camera_cone_height

    opengl_coord_transform = np.eye(4)
    opengl_coord_transform[1, 1] = -1
    opengl_coord_transform[2, 2] = -1

    final_transform = camera_transform @ opengl_coord_transform @ z_rotation_matrix

    minor_rotation = np.eye(4)
    minor_rotation[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()

    original_vertices = base_cone.vertices
    scaled_vertices = 0.95 * original_vertices
    rotated_vertices = apply_transformation_to_points(minor_rotation, original_vertices)

    all_vertices = np.concatenate([original_vertices, scaled_vertices, rotated_vertices])
    transformed_vertices = apply_transformation_to_points(final_transform, all_vertices)

    camera_faces = generate_camera_mesh_faces(base_cone)

    camera_mesh = trimesh.Trimesh(vertices=transformed_vertices, faces=camera_faces)
    camera_mesh.visual.face_colors[:, :3] = camera_color

    scene.add_geometry(camera_mesh)


def build_composite_scene(
    points3d: np.ndarray,
    images: np.ndarray,
    camera_poses: np.ndarray,
    normals: np.ndarray = None,
    valid_masks: np.ndarray = None,
    show_camera: bool = True,
    as_mesh: bool = True,
) -> trimesh.Scene:
    """
    Args:
        points3d: [S, H, W, 3] world-space points, one per frame
        images: [S, H, W, 3] float [0,1] source images, same H,W as points3d
        camera_poses: [S, 4, 4] camera-to-world matrices
        normals: optional [S, H, W, 3]
        valid_masks: optional [S, H, W] bool keep-masks (e.g. combined sky + edge mask)
        show_camera: add a color-coded frustum marker per view
        as_mesh: per-frame textured mesh (True) or a single merged point cloud (False)

    Returns:
        trimesh.Scene ready to `.export(path, file_type='glb')`
    """
    num_frames = points3d.shape[0]

    all_vertices_for_scale = points3d.reshape(-1, 3)
    percentile_lower = np.percentile(all_vertices_for_scale, 5, axis=0)
    percentile_upper = np.percentile(all_vertices_for_scale, 95, axis=0)
    scene_scale_factor = np.linalg.norm(percentile_upper - percentile_lower)
    if not np.isfinite(scene_scale_factor) or scene_scale_factor <= 0:
        scene_scale_factor = 1.0

    color_palette = __import__("matplotlib").colormaps["gist_rainbow"]  # .get_cmap() removed in newer matplotlib
    output_scene = trimesh.Scene()

    if as_mesh:
        for frame_idx in range(num_frames):
            frame_points = points3d[frame_idx] * np.array([1, -1, 1], dtype=np.float32)
            frame_colors = images[frame_idx].astype(np.float32)
            frame_mask = valid_masks[frame_idx] if valid_masks is not None else None

            if normals is not None:
                frame_normals = normals[frame_idx] * np.array([1, -1, 1], dtype=np.float32)
                faces, verts, cols, norms = create_image_mesh(
                    frame_points, frame_colors, frame_normals,
                    mask=frame_mask, triangulate=True, return_vertex_indices=False,
                )
            else:
                faces, verts, cols = create_image_mesh(
                    frame_points, frame_colors,
                    mask=frame_mask, triangulate=True, return_vertex_indices=False,
                )
                norms = None

            if len(verts) == 0 or len(faces) == 0:
                continue

            geometry_mesh = trimesh.Trimesh(
                vertices=verts,
                faces=faces,
                vertex_colors=(np.clip(cols, 0, 1) * 255).astype(np.uint8),
                vertex_normals=norms,
                process=False,
            )
            output_scene.add_geometry(geometry_mesh)
    else:
        flat_vertices = points3d.reshape(-1, 3)
        flat_colors = (np.clip(images, 0, 1).reshape(-1, 3) * 255).astype(np.uint8)
        if valid_masks is not None:
            keep = valid_masks.reshape(-1)
            flat_vertices = flat_vertices[keep]
            flat_colors = flat_colors[keep]
        output_scene.add_geometry(trimesh.PointCloud(vertices=flat_vertices, colors=flat_colors))

    if show_camera:
        for camera_idx in range(num_frames):
            camera_color_rgba = color_palette(camera_idx / max(num_frames, 1))
            camera_color_rgb = tuple(int(255 * x) for x in camera_color_rgba[:3])
            integrate_camera_into_scene(output_scene, camera_poses[camera_idx], camera_color_rgb, scene_scale_factor)

    opengl_transform = np.eye(4)
    opengl_transform[1, 1] = -1
    opengl_transform[2, 2] = -1
    scene_transformation = np.linalg.inv(camera_poses[0]) @ opengl_transform
    output_scene.apply_transform(scene_transformation)

    return output_scene
