"""Build the hero animation assets from a RealNav scene and its capture.

The hero shows the scene from one fixed bird's-eye camera. Every 3D Gaussian
is projected offline into a 2D Gaussian of that view: screen position, 2D
covariance and a view-dependent color evaluated from its spherical harmonics.
Gaussians the view cannot see are pruned. The visible ones are rasterized
into one image, and the strongest of them are also written as 2D splats in
back-to-front order: the page grows those splats at the reveal wavefront and
fades the full image in behind it.

The scan replays the capture the Gaussians were trained on: the training
views of the PGSR run (cameras.json, in the Gaussian world frame) give the
camera path and the poses of the photos, the Stray Scanner capture gives
their video frames, and its LiDAR depth tells when each scene point first
came into view.

Usage: python tools/build_hero.py <scene_dir> --capture <stray_dir>
       --cameras <cameras.json> [--out static/hero] [--cache .cache/hero]
"""

import argparse
import gzip
import io
import json
import math
import subprocess
from itertools import pairwise
from pathlib import Path

import cv2
import h5py
import numba
import numpy as np
from PIL import Image

# Reference frame of the projected scene, in pixels.
WIDTH, HEIGHT = 2560, 1440
VERTICAL_FOV_DEG = 50.0
TILT_DEG = 22.0  # Camera tilt from straight down, toward +Z.
CEILING_M = 1.7  # Gaussians above this aligned height are cut away.
WALL_M = 0.22  # Gaussians farther than this outside the walkable outline are cut away.
MARGIN = 0.05  # Fraction of the frame kept free around the scene.
SH_DEGREE = 3
SH_MAX_ELEVATION_DEG = 30.0  # Steepest view elevation used to evaluate the harmonics.
LOW_PASS_PX2 = 0.3  # Screen-space dilation of every 2D covariance.
WAVE_SPLATS = 70000  # Splats that grow at the reveal wavefront.
LOG2_SIGMA_RANGE = (-4.0, 9.0)

VIEW_SIZE = (400, 300)  # Cached video frames of the training views.
PHOTO_COUNT = 54  # Photos of the scan, one per equal slice of its time.
PHOTO_TILE = (320, 240)
PHOTO_COLUMNS = 6
PATH_STEP = 2  # Training views per sample of the scan path.
POINT_CELL_PX = 10  # One scan point at a random spot of each cell of this size in the reference frame.
POSITION_SCALE = (
    0.5  # Resolution of the rendered world positions relative to the reference frame.
)
DEPTH_RANGE_M = (0.15, 3.0)  # LiDAR depth that counts a point as seen.
DEPTH_TOLERANCE_M = 0.15
SHOWCASE = ["000068", "000100", "000148", "000147", "000012", "000235"]

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
SH_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)


def read_ply(path: Path) -> dict[str, np.ndarray]:
    """Read a binary little-endian 3DGS PLY of float32 properties."""
    with open(path, "rb") as file:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += file.readline()
        lines = header.decode().splitlines()
        count = next(
            int(line.split()[-1]) for line in lines if line.startswith("element vertex")
        )
        names = [line.split()[-1] for line in lines if line.startswith("property")]
        data = np.fromfile(file, dtype=np.float32, count=count * len(names)).reshape(
            count, len(names)
        )
    return {name: data[:, index] for index, name in enumerate(names)}


def quaternion_matrices(q: np.ndarray) -> np.ndarray:
    """Return (N, 3, 3) rotations of (N, 4) wxyz quaternions."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        axis=1,
    ).reshape(-1, 3, 3)


def sh_colors(
    ply: dict[str, np.ndarray], directions: np.ndarray, degree: int
) -> np.ndarray:
    """Evaluate spherical harmonics up to degree (N, 3) along unit view directions (N, 3)."""
    dc = np.stack([ply[f"f_dc_{c}"] for c in range(3)], axis=1)
    rest = np.stack([ply[f"f_rest_{i}"] for i in range(45)], axis=1).reshape(-1, 3, 15)
    x, y, z = directions.T
    xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
    basis = np.stack(
        [
            -SH_C1 * y,
            SH_C1 * z,
            -SH_C1 * x,
            SH_C2[0] * xy,
            SH_C2[1] * yz,
            SH_C2[2] * (2 * zz - xx - yy),
            SH_C2[3] * xz,
            SH_C2[4] * (xx - yy),
            SH_C3[0] * y * (3 * xx - yy),
            SH_C3[1] * xy * z,
            SH_C3[2] * y * (4 * zz - xx - yy),
            SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy),
            SH_C3[4] * x * (4 * zz - xx - yy),
            SH_C3[5] * z * (xx - yy),
            SH_C3[6] * x * (xx - 3 * yy),
        ],
        axis=1,
    )
    terms = (degree + 1) ** 2 - 1
    return np.clip(
        SH_C0 * dc
        + np.einsum("nck,nk->nc", rest[:, :, :terms], basis[:, :terms])
        + 0.5,
        0.0,
        1.0,
    )


class Camera:
    """A pinhole camera in the aligned frame with OpenCV axes: x right, y down, z forward."""

    def __init__(self, target: np.ndarray, distance: float):
        tilt = math.radians(TILT_DEG)
        self.eye = target + distance * np.array([0.0, math.cos(tilt), math.sin(tilt)])
        forward = (target - self.eye) / np.linalg.norm(target - self.eye)
        right = np.cross(forward, [0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        self.rotation = np.stack([right, down, forward])
        self.focal = HEIGHT / (2.0 * math.tan(math.radians(VERTICAL_FOV_DEG) / 2.0))

    def to_camera(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points, np.float64) - self.eye) @ self.rotation.T

    def project(self, points: np.ndarray) -> np.ndarray:
        """Return (N, 3) pixel coordinates and depth of aligned points (N, 3)."""
        cam = self.to_camera(points)
        return np.column_stack(
            [
                self.focal * cam[:, 0] / cam[:, 2] + WIDTH / 2,
                self.focal * cam[:, 1] / cam[:, 2] + HEIGHT / 2,
                cam[:, 2],
            ]
        )

    def manifest(self) -> dict:
        return {
            "eye": self.eye.round(6).tolist(),
            "rotation": self.rotation.round(8).ravel().tolist(),
            "focal": round(self.focal, 4),
        }


def fit_camera(points: np.ndarray, target: np.ndarray) -> Camera:
    """Return the closest camera whose frame holds all points within the margin."""
    low, high = 2.0, 60.0
    for _ in range(40):
        distance = (low + high) / 2
        uv = Camera(target, distance).project(points)
        inside = (
            (uv[:, 0] >= MARGIN * WIDTH).all()
            and (uv[:, 0] <= (1 - MARGIN) * WIDTH).all()
            and (uv[:, 1] >= MARGIN * HEIGHT).all()
            and (uv[:, 1] <= (1 - MARGIN) * HEIGHT).all()
        )
        low, high = (low, distance) if inside else (distance, high)
    return Camera(target, high)


def visibility(uv: np.ndarray, sigma: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Return each Gaussian's best transmittance-weighted alpha over the cells it covers.

    Gaussians arrive front to back and are visited in batches; each covers the
    cells of a 2-pixel grid within one standard deviation, and each batch sees
    the transmittance left by earlier batches.
    """
    cell = 2.0
    grid_w, grid_h = int(WIDTH / cell) + 1, int(HEIGHT / cell) + 1
    log_t = np.zeros(grid_w * grid_h)
    best = np.zeros(len(uv))
    radius = np.clip(np.ceil(sigma / cell).astype(int), 0, 8)
    batch = 20000
    for start in range(0, len(uv), batch):
        index = np.arange(start, min(start + batch, len(uv)))
        cells, owners = [], []
        for r in np.unique(radius[index]):
            members = index[radius[index] == r]
            offsets = np.arange(-r, r + 1)
            ox, oy = np.meshgrid(offsets, offsets)
            disk = ox.ravel() ** 2 + oy.ravel() ** 2 <= max(r, 0.5) ** 2
            ox, oy = ox.ravel()[disk], oy.ravel()[disk]
            cx = (uv[members, 0] / cell).astype(int)[:, None] + ox
            cy = (uv[members, 1] / cell).astype(int)[:, None] + oy
            valid = (cx >= 0) & (cx < grid_w) & (cy >= 0) & (cy < grid_h)
            cells.append((cy * grid_w + cx)[valid])
            owners.append(np.broadcast_to(members[:, None], cx.shape)[valid])
        cells, owners = np.concatenate(cells), np.concatenate(owners)
        np.maximum.at(best, owners, np.exp(log_t[cells]) * alpha[owners])
        np.add.at(log_t, cells, np.log1p(-np.minimum(alpha[owners] * 0.6, 0.99)))
    return best


@numba.njit(cache=True)
def rasterize(x, y, s1, s2, theta, rgba, width, height):
    """Composite 2D Gaussians in the given back-to-front order, as the page shader does."""
    image = np.zeros((height, width, 4), np.float32)
    for i in range(len(x)):
        c, s = math.cos(theta[i]), math.sin(theta[i])
        v1, v2 = s1[i] * s1[i] + LOW_PASS_PX2, s2[i] * s2[i] + LOW_PASS_PX2
        extent = 3.0 * math.sqrt(v1)
        x0, x1 = max(0, int(x[i] - extent)), min(width - 1, int(x[i] + extent) + 1)
        y0, y1 = max(0, int(y[i] - extent)), min(height - 1, int(y[i] + extent) + 1)
        for py in range(y0, y1 + 1):
            dy = py + 0.5 - y[i]
            for px in range(x0, x1 + 1):
                dx = px + 0.5 - x[i]
                u, w = c * dx + s * dy, -s * dx + c * dy
                m = u * u / v1 + w * w / v2
                if m > 9.0:
                    continue
                a = min(rgba[i, 3] * math.exp(-0.5 * m), 0.99)
                if a < 1.0 / 255.0:
                    continue
                for k in range(3):
                    image[py, px, k] = image[py, px, k] * (1.0 - a) + rgba[i, k] * a
                image[py, px, 3] = image[py, px, 3] * (1.0 - a) + a
    return image


def encode_splats(uv, sigma_major, sigma_minor, angle, rgba) -> bytes:
    """Return gzip bytes of the columnar splat planes in draw order."""
    x = np.clip(np.round(uv[:, 0] / WIDTH * 65535), 0, 65535).astype("<u2")
    y = np.clip(np.round(uv[:, 1] / HEIGHT * 65535), 0, 65535).astype("<u2")
    lo, hi = LOG2_SIGMA_RANGE
    s1 = np.clip(
        np.round((np.log2(sigma_major) - lo) / (hi - lo) * 255), 0, 255
    ).astype(np.uint8)
    s2 = np.clip(
        np.round((np.log2(sigma_minor) - lo) / (hi - lo) * 255), 0, 255
    ).astype(np.uint8)
    th = (
        np.round(np.mod(angle, math.pi) / math.pi * 256).astype(np.int64) % 256
    ).astype(np.uint8)
    color = np.clip(np.round(rgba * 255), 0, 255).astype(np.uint8)
    planes = [
        x.view(np.uint8)[0::2],
        x.view(np.uint8)[1::2],
        y.view(np.uint8)[0::2],
        y.view(np.uint8)[1::2],
        s1,
        s2,
        th,
        color[:, 0],
        color[:, 1],
        color[:, 2],
        color[:, 3],
    ]
    return gzip.compress(
        b"".join(p.tobytes() for p in planes), compresslevel=9, mtime=0
    )


def save_webp(
    image: np.ndarray, path: Path, size: tuple[int, int], quality: int
) -> None:
    """Save a straight-alpha float RGBA image as lossy WebP."""
    rgba = image.copy()
    alpha = rgba[..., 3:4]
    rgba[..., :3] = np.where(alpha > 1e-4, rgba[..., :3] / np.maximum(alpha, 1e-4), 0)
    pil = Image.fromarray(np.clip(rgba * 255 + 0.5, 0, 255).astype(np.uint8), "RGBA")
    pil.resize(size, Image.LANCZOS).save(
        path, "WEBP", quality=quality, method=6, alpha_quality=80
    )


def footprint_mask(scene_dir: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the BEV area inside the outer walls, grown by WALL_M, with its origin and resolution."""
    with h5py.File(scene_dir / "scene.h5") as file:
        raw = file["bev/raw"][...]
        origin = file["bev/origin_xz"][...]
        resolution = float(file["bev/resolution"][()])
    blocked = np.pad((raw == 0).astype(np.uint8), 1, constant_values=1)
    flood = blocked.copy()
    cv2.floodFill(flood, None, (0, 0), 2)
    inside = (flood[1:-1, 1:-1] != 2).astype(np.uint8)
    grow = round(float(WALL_M / resolution))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    return cv2.dilate(inside, kernel) > 0, origin, resolution


def build_scene(
    scene_dir: Path, transform: np.ndarray, out: Path
) -> tuple[Camera, list[int], dict]:
    ply = read_ply(scene_dir / "gaussians.ply")
    world = np.stack([ply["x"], ply["y"], ply["z"]], axis=1).astype(np.float64)
    rotation_align = transform[:3, :3]
    points = world @ rotation_align.T + transform[:3, 3]
    opacity = 1.0 / (1.0 + np.exp(-ply["opacity"].astype(np.float64)))
    scales = np.exp(
        np.stack([ply[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float64)
    )

    mask, origin, resolution = footprint_mask(scene_dir)
    col = np.floor((points[:, 0] - origin[0]) / resolution).astype(int)
    row = np.floor((points[:, 2] - origin[1]) / resolution).astype(int)
    on_map = (col >= 0) & (col < mask.shape[1]) & (row >= 0) & (row < mask.shape[0])
    inside = np.zeros(len(points), bool)
    inside[on_map] = mask[row[on_map], col[on_map]]
    keep = (
        inside
        & (opacity > 0.02)
        & (points[:, 1] > -0.15)
        & (points[:, 1] < CEILING_M)
        & (scales.max(1) < 0.5)
    )
    low = points[keep][:, [0, 2]].min(0)
    high = points[keep][:, [0, 2]].max(0)
    target = np.array([(low[0] + high[0]) / 2, 0.0, (low[1] + high[1]) / 2])
    corners = np.array(
        [
            [x, h, z]
            for x in (low[0], high[0])
            for z in (low[1], high[1])
            for h in (0.0, CEILING_M)
        ]
    )
    camera = fit_camera(corners, target)

    index = np.where(keep)[0]
    proj = camera.project(points[index])
    cam = camera.to_camera(points[index])
    margin = 64
    in_frame = (
        (proj[:, 2] > 0.1)
        & (proj[:, 0] > -margin)
        & (proj[:, 0] < WIDTH + margin)
        & (proj[:, 1] > -margin)
        & (proj[:, 1] < HEIGHT + margin)
    )
    index, proj, cam = index[in_frame], proj[in_frame], cam[in_frame]

    # 3D covariance in camera axes, then the EWA projection to pixels.
    rot = quaternion_matrices(
        np.stack([ply[f"rot_{i}"] for i in range(4)], axis=1)[index].astype(np.float64)
    )
    m = (camera.rotation @ rotation_align)[None] @ rot * scales[index][:, None, :]
    cov = m @ m.transpose(0, 2, 1)
    f, x, y, z = camera.focal, cam[:, 0], cam[:, 1], cam[:, 2]
    jac = np.zeros((len(index), 2, 3))
    jac[:, 0, 0], jac[:, 0, 2] = f / z, -f * x / z**2
    jac[:, 1, 1], jac[:, 1, 2] = f / z, -f * y / z**2
    cov2 = jac @ cov @ jac.transpose(0, 2, 1)
    a, b, c = cov2[:, 0, 0], cov2[:, 0, 1], cov2[:, 1, 1]
    mid, rad = (a + c) / 2, np.sqrt(((a - c) / 2) ** 2 + b**2)
    sigma_major = np.sqrt(np.maximum(mid + rad, 1e-6))
    sigma_minor = np.sqrt(np.maximum(mid - rad, 1e-6))
    angle = 0.5 * np.arctan2(2 * b, a - c)

    # The scan looks roughly level, so higher harmonics extrapolate badly toward a
    # bird's-eye view; evaluate them along the view azimuth at a limited elevation.
    directions = points[index] - camera.eye
    directions[:, 1] = np.maximum(
        directions[:, 1],
        -np.tan(math.radians(SH_MAX_ELEVATION_DEG))
        * np.linalg.norm(directions[:, [0, 2]], axis=1),
    )
    directions = directions @ rotation_align  # Aligned to original frame.
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    rgb = sh_colors({k: v[index] for k, v in ply.items()}, directions, SH_DEGREE)
    rgba = np.column_stack([rgb, opacity[index]]).astype(np.float32)

    # Weight sub-pixel splats by their area so that they cannot shadow the visibility test.
    footprint = np.minimum(1.0, sigma_major * sigma_minor / 0.25) * opacity[index]
    front_to_back = np.argsort(proj[:, 2])
    best = np.zeros(len(index))
    best[front_to_back] = visibility(
        proj[front_to_back, :2], sigma_major[front_to_back], footprint[front_to_back]
    )
    visible = np.where(best >= 0.01)[0]
    print(
        f"gaussians: {len(ply['x'])} -> in view {len(index)} -> visible {len(visible)}"
    )

    draw = visible[np.argsort(-proj[visible, 2])]
    image = rasterize(
        proj[draw, 0],
        proj[draw, 1],
        sigma_major[draw],
        sigma_minor[draw],
        angle[draw],
        rgba[draw],
        WIDTH,
        HEIGHT,
    )
    ys, xs = np.nonzero(image[..., 3] > 0.1)
    content_box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    save_webp(image, out / "scene.webp", (WIDTH, HEIGHT), 84)
    save_webp(image, out / "scene-sm.webp", (WIDTH * 5 // 8, HEIGHT * 5 // 8), 82)

    score = best * np.minimum(sigma_major * sigma_minor, 64.0)
    wave = visible[np.argsort(-score[visible])[:WAVE_SPLATS]]
    wave = wave[np.argsort(-proj[wave, 2])]
    blob = encode_splats(
        proj[wave, :2], sigma_major[wave], sigma_minor[wave], angle[wave], rgba[wave]
    )
    (out / "splats.bin").write_bytes(blob)
    print(f"wave splats: {len(wave)}, {len(blob) / 1e6:.2f} MB")
    # World positions composited like colors: each pixel's alpha-weighted mean of the Gaussian centers.
    k = POSITION_SCALE
    weighted = np.column_stack([world[index[draw]], opacity[index[draw]]]).astype(
        np.float32
    )
    position = rasterize(
        proj[draw, 0] * k,
        proj[draw, 1] * k,
        sigma_major[draw] * k,
        sigma_minor[draw] * k,
        angle[draw],
        weighted,
        int(WIDTH * k),
        int(HEIGHT * k),
    )
    return camera, content_box, {"color": image, "position": position}


def rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Simplify a polyline with the Ramer-Douglas-Peucker algorithm."""
    if len(points) < 3:
        return points
    start, end = points[0], points[-1]
    line = end - start
    norm = np.linalg.norm(line)
    rel = points - start
    dist = (
        np.abs(line[0] * rel[:, 1] - line[1] * rel[:, 0]) / norm
        if norm > 1e-9
        else np.linalg.norm(rel, axis=1)
    )
    index = int(np.argmax(dist))
    if dist[index] <= epsilon:
        return np.stack([start, end])
    return np.concatenate(
        [rdp(points[: index + 1], epsilon)[:-1], rdp(points[index:], epsilon)]
    )


def jpeg_frames(group: h5py.Group) -> list[bytes]:
    data, offsets = group["data"][...], group["offsets"][...]
    return [
        data[offsets[i] : offsets[i + 1]].tobytes() for i in range(len(offsets) - 1)
    ]


def decode(jpeg: bytes) -> Image.Image:
    return Image.open(io.BytesIO(jpeg)).convert("RGB")


def build_episodes(scene_dir: Path, out: Path) -> dict:
    """Write the goal views of the showcased episodes; return every episode's path and instruction."""
    with h5py.File(scene_dir / "trajectories.h5") as file:
        group = file["trajectories"]
        keys = sorted(group)
        episodes = []
        for key in keys:
            g = group[key]
            episodes.append(
                {
                    "id": key,
                    "type": g["instruction/type"][()].decode(),
                    "text": g["instruction/text"][()].decode(),
                    "path": np.round(rdp(g["path_xz"][...], 0.02), 2).ravel().tolist(),
                }
            )
        goal_w, goal_h = 280, 210
        goals = Image.new("RGB", (goal_w, goal_h * len(SHOWCASE)))
        for i, key in enumerate(SHOWCASE):
            goal = decode(jpeg_frames(group[f"{key}/instruction_images/goal/rgb"])[0])
            goals.paste(goal.resize((goal_w, goal_h), Image.LANCZOS), (0, i * goal_h))
        goals.save(out / "goals.webp", "WEBP", quality=66, method=6)
    return {
        "episodes": episodes,
        "showcase": [keys.index(key) for key in SHOWCASE],
        "goalTile": [goal_w, goal_h],
    }


class Views:
    """The training views of the Gaussians in capture order, read from PGSR cameras.json."""

    def __init__(self, path: Path, transform: np.ndarray):
        cameras = sorted(
            json.loads(path.read_text()), key=lambda camera: int(camera["img_name"])
        )
        self.frames = np.array(
            [int(camera["img_name"]) for camera in cameras]
        )  # Video frame indices.
        self.position = np.array(
            [camera["position"] for camera in cameras]
        )  # (N, 3) world centers.
        self.rotation = np.array(
            [camera["rotation"] for camera in cameras]
        )  # (N, 3, 3) camera to world.
        rotation = transform[:3, :3]
        self.aligned = self.position @ rotation.T + transform[:3, 3]
        forward = self.rotation[:, :, 2] @ rotation.T
        forward[:, 1] = 0.0
        self.forward = forward / np.linalg.norm(
            forward, axis=1, keepdims=True
        )  # Level view directions.

    def __len__(self) -> int:
        return len(self.frames)

    def image_poses(
        self, camera: Camera, indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the reference-frame positions (N, 2) and view direction angles (N,) of views."""
        at = camera.project(self.aligned[indices])[:, :2]
        ahead = camera.project(self.aligned[indices] + 0.4 * self.forward[indices])[
            :, :2
        ]
        return at, np.arctan2(ahead[:, 1] - at[:, 1], ahead[:, 0] - at[:, 0])


def extract_views(video: Path, views: Views, cache: Path) -> list[Path]:
    """Decode the video frames of the views once into VIEW_SIZE JPEGs under cache/views."""
    directory = cache / "views"
    paths = [directory / f"{i:05d}.jpg" for i in range(len(views))]
    if all(path.exists() for path in paths):
        return paths
    step = int(views.frames[1] - views.frames[0])
    if not np.array_equal(views.frames, np.arange(len(views)) * step):
        raise ValueError(
            "The training views must be every step-th video frame from frame 0"
        )
    directory.mkdir(parents=True, exist_ok=True)
    width, height = VIEW_SIZE
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-hwaccel",
            "auto",
            "-i",
            str(video),
            "-vf",
            f"select=not(mod(n\\,{step})),scale={width}:{height}:flags=area",
            "-vsync",
            "0",
            "-q:v",
            "3",
            "-frames:v",
            str(len(views)),
            "-start_number",
            "0",
            str(directory / "%05d.jpg"),
        ],
        check=True,
    )
    return paths


def choose_photos(views: Views, paths: list[Path]) -> list[int]:
    """Pick one sharp view per equal slice of the scan, preferring poses unlike the earlier picks."""
    sharpness = np.array(
        [
            cv2.Laplacian(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), cv2.CV_64F).var()
            for path in paths
        ]
    )
    heading = np.arctan2(views.forward[:, 0], -views.forward[:, 2])
    features = np.column_stack(
        [
            views.aligned[:, 0],
            views.aligned[:, 2],
            0.8 * np.cos(heading),
            0.8 * np.sin(heading),
        ]
    )
    edges = np.linspace(0, len(views), PHOTO_COUNT + 1).astype(int)
    chosen = []
    for begin, end in pairwise(edges):
        candidates = np.arange(begin, end)
        rank = sharpness[candidates].argsort().argsort() / max(1, len(candidates) - 1)
        novelty = np.ones(len(candidates))
        if chosen:
            novelty = np.linalg.norm(
                features[candidates, None] - features[chosen][None], axis=2
            ).min(1)
        chosen.append(
            int(candidates[np.argmax(novelty * (0.35 + rank) * (rank >= 0.3))])
        )
    return chosen


def first_seen(
    points: np.ndarray, views: Views, capture: Path, size: tuple[int, int]
) -> np.ndarray:
    """Return the first view whose LiDAR depth sees each world point (N, 3), or -1."""
    rows = [
        line.replace(",", " ").split()
        for line in (capture / "camera_matrix.csv").read_text().splitlines()
        if line.strip()
    ]
    fx, fy, cx, cy = (
        float(rows[0][0]),
        float(rows[1][1]),
        float(rows[0][2]),
        float(rows[1][2]),
    )
    width, height = size
    first = np.full(len(points), -1)
    for j, frame in enumerate(views.frames):
        todo = np.flatnonzero(first < 0)
        depth = cv2.imread(
            str(capture / "depth" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED
        )
        if not len(todo) or depth is None:
            continue
        cam = (points[todo] - views.position[j]) @ views.rotation[j]
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u, v = fx * cam[:, 0] / z + cx, fy * cam[:, 1] / z + cy
        inside = np.flatnonzero(
            (z > DEPTH_RANGE_M[0])
            & (z < DEPTH_RANGE_M[1])
            & (u >= 0)
            & (u < width)
            & (v >= 0)
            & (v < height)
        )
        measured = (
            depth[
                (v[inside] * depth.shape[0] / height).astype(int),
                (u[inside] * depth.shape[1] / width).astype(int),
            ]
            / 1000.0
        )
        hit = (measured > 0) & (z[inside] < measured + DEPTH_TOLERANCE_M)
        first[todo[inside[hit]]] = j
    return first


def build_scan(
    views: Views, capture: Path, cache: Path, camera: Camera, surface: dict, out: Path
) -> dict:
    """Write the photos and the scan points; return the scan path and the photo poses."""
    video = capture / "rgb.mp4"
    reader = cv2.VideoCapture(str(video))
    size = (
        int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    duration = reader.get(cv2.CAP_PROP_FRAME_COUNT) / reader.get(cv2.CAP_PROP_FPS)
    reader.release()
    paths = extract_views(video, views, cache)

    # Photos, one per slice of the scan time, in an atlas of PHOTO_TILE tiles.
    chosen = choose_photos(views, paths)
    tile_w, tile_h = PHOTO_TILE
    atlas = Image.new(
        "RGB", (tile_w * PHOTO_COLUMNS, tile_h * math.ceil(PHOTO_COUNT / PHOTO_COLUMNS))
    )
    for slot, i in enumerate(chosen):
        image = Image.open(paths[i]).convert("RGB").resize(PHOTO_TILE, Image.LANCZOS)
        atlas.paste(
            image, ((slot % PHOTO_COLUMNS) * tile_w, (slot // PHOTO_COLUMNS) * tile_h)
        )
    atlas.save(out / "photos.webp", "WEBP", quality=46, method=6)
    at, angle = views.image_poses(camera, np.array(chosen))
    photos = [
        [i, round(float(u), 1), round(float(v), 1), round(float(a), 3)]
        for i, (u, v), a in zip(chosen, at, angle)
    ]

    # Camera path, every PATH_STEP-th view, as integer pixels and centiradians.
    at, angle = views.image_poses(camera, np.arange(0, len(views), PATH_STEP))
    path = (
        np.column_stack([np.round(at), np.round(angle * 100)])
        .astype(int)
        .ravel()
        .tolist()
    )

    # Scan points: one surface sample per cell, appearing when the LiDAR first sees it.
    rng = np.random.default_rng(0)
    grid_x, grid_y = np.meshgrid(
        np.arange(0, WIDTH, POINT_CELL_PX), np.arange(0, HEIGHT, POINT_CELL_PX)
    )
    uv = (
        np.column_stack([grid_x.ravel(), grid_y.ravel()])
        + rng.random((grid_x.size, 2)) * POINT_CELL_PX
    )
    pixel = np.minimum(uv.astype(int), [WIDTH - 1, HEIGHT - 1])
    color = surface["color"][pixel[:, 1], pixel[:, 0]]
    covered = np.flatnonzero(color[:, 3] > 0.6)
    sample = (pixel[covered] * POSITION_SCALE).astype(int)
    position = surface["position"][sample[:, 1], sample[:, 0]]
    seen = first_seen(
        position[:, :3] / np.maximum(position[:, 3:], 1e-6), views, capture, size
    )
    pick = covered[seen >= 0]
    seen = seen[seen >= 0]
    rgb = color[pick, :3] / color[pick, 3:]
    x = np.clip(np.round(uv[pick, 0] / WIDTH * 65535), 0, 65535).astype("<u2")
    y = np.clip(np.round(uv[pick, 1] / HEIGHT * 65535), 0, 65535).astype("<u2")
    color = np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8)
    time = np.round(seen / (len(views) - 1) * 255).astype(np.uint8)
    planes = [
        x.view(np.uint8)[0::2],
        x.view(np.uint8)[1::2],
        y.view(np.uint8)[0::2],
        y.view(np.uint8)[1::2],
    ]
    planes += [color[:, 0], color[:, 1], color[:, 2], time]
    blob = gzip.compress(
        b"".join(plane.tobytes() for plane in planes), compresslevel=9, mtime=0
    )
    (out / "scan.bin").write_bytes(blob)
    print(f"scan points: {len(pick)} of {len(covered)} seen, {len(blob) / 1e3:.0f} kB")
    return {
        "scan": {
            "duration": round(duration, 2),
            "views": len(views),
            "pathStep": PATH_STEP,
            "path": path,
        },
        "photos": photos,
        "photoTile": [tile_w, tile_h, PHOTO_COLUMNS],
    }


def build_map(scene_dir: Path, camera: Camera, out: Path) -> dict:
    """Warp the navigable BEV map onto the floor plane of the view."""
    with h5py.File(scene_dir / "scene.h5") as file:
        navigable = file["bev/navigable"][...]
        origin = file["bev/origin_xz"][...]
        resolution = float(file["bev/resolution"][()])
    rows, cols = navigable.shape
    src = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
    world = np.array(
        [[origin[0] + u * resolution, 0.0, origin[1] + v * resolution] for u, v in src]
    )
    scale = 0.5
    dst = (camera.project(world)[:, :2] * scale).astype(np.float32)
    homography = cv2.getPerspectiveTransform(src, dst)
    size = (int(WIDTH * scale), int(HEIGHT * scale))
    warped = cv2.warpPerspective(navigable, homography, size, flags=cv2.INTER_LINEAR)
    Image.fromarray(warped, "L").save(out / "map.png", optimize=True)
    return {"mapScale": scale}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument(
        "--capture",
        type=Path,
        required=True,
        help="Stray Scanner capture with rgb.mp4 and depth/",
    )
    parser.add_argument(
        "--cameras",
        type=Path,
        required=True,
        help="cameras.json of the PGSR training run",
    )
    parser.add_argument("--out", type=Path, default=Path("static/hero"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/hero"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.scene_dir / "scene.h5") as file:
        transform = file["alignment/world_to_aligned"][...]
    camera, content_box, surface = build_scene(args.scene_dir, transform, args.out)
    manifest = {
        "width": WIDTH,
        "height": HEIGHT,
        "contentBox": content_box,
        "camera": camera.manifest(),
        "waveSplats": WAVE_SPLATS,
        "log2Sigma": LOG2_SIGMA_RANGE,
        "lowPass": LOW_PASS_PX2,
    }
    manifest |= build_map(args.scene_dir, camera, args.out)
    manifest |= build_episodes(args.scene_dir, args.out)
    manifest |= build_scan(
        Views(args.cameras, transform),
        args.capture,
        args.cache,
        camera,
        surface,
        args.out,
    )
    (args.out / "hero.json").write_text(json.dumps(manifest, separators=(",", ":")))
    for path in sorted(args.out.iterdir()):
        print(f"{path.name:14s} {path.stat().st_size / 1e3:8.1f} kB")


if __name__ == "__main__":
    main()
