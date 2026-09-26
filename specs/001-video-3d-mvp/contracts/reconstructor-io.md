# Contract: reconstructor input and output

**Status: the contract was designed from upstream documentation. It has not been run,
compatibility has not been verified.**

Our code **does not call** the reconstructor and does not import it. The contract describes the
files that cross the boundary.

## 1. What the reconstructor must accept

A directory with an `images/` subdirectory containing .jpg frames of the same orientation.
Our package imposes no other requirements.

## 2. What it must return (one format for all variants)

| File | Required | Content |
|---|---|---|
| `sparse/cameras.bin` | yes | camera models and intrinsics |
| `sparse/images.bin` | yes | poses (quaternion + t, **world→camera**) and image name |
| `sparse/points3D.bin` | yes | 3D points with RGB color |
| `sparse/points.ply` | no | the upstream colored point cloud. **Verified 2026-09-20**: the file is written inside `sparse/`, not into the directory root. We do not need it — the normalized point cloud is built from `points3D.bin` |
| run log | desirable | upstream stdout, saved as is |

The COLMAP sparse format is chosen as the single intake point: both the primary and the fallback
variant produce it, so there is **one** reader on the Mac side.

## 3. Variant A (primary): VGGT

```bash
python demo_colmap.py --scene_dir=<package>            # without BA
python demo_colmap.py --scene_dir=<package> --use_ba   # with BA
```

Information from upstream documentation (not verified by a run):
- expected input layout — `scene_dir/images/`;
- output — `scene_dir/sparse/{cameras,images,points3D}.bin` and `sparse/points.ply`;
- defaults: `--seed 42`, `--max_reproj_error 8.0`,
  `--camera_type SIMPLE_PINHOLE`, `--vis_thresh 0.2`, `--query_frame_num 8`,
  `--max_query_pts 4096`, `--conf_thres_value 5.0`, `--fine_tracking` enabled;
- VGGT's internal processing resolution is 518 px; image loading in the script uses 1024 px;
- convention — OpenCV, "camera from world";
- weights used: `facebook/VGGT-1B`, license **CC-BY-NC-4.0** (non-commercial
  use). The restriction is recorded in the project README.

**The confidence threshold is a critical parameter (measured at M1, 2026-09-20).** In the branch
without bundle adjustment, points are taken from depth maps and filtered by `conf_mask = depth_conf >=
conf_thres_value`. The default value **5.0 gives zero points** on our material with fully
recovered cameras, and the script does not report this as an error. The measured dependence for
`pumpkin360.mp4`:

| Threshold | Points |
|---|---|
| 5.0 | 0 |
| 1.8 | ≈ 9 300 |
| 1.6 | > 100 000 (hits the script's `max_points_for_colmap` limit) |
| 1.3 | > 100 000 |
| 1.0 | > 100 000 |

The working value for footage of this kind is **1.6**: the object is recognizable, the background
has practically gone. The threshold MUST be recorded in the run manifest: without it the result
is not reproducible.

**What we record in the manifest about the reconstructor**: variant name, repository commit/version,
weights identifier and revision, the full command line with all arguments, `--use_ba` yes/no,
torch and CUDA versions, `nvidia-smi` output (GPU model, VRAM), execution time.

## 4. Variant B (fallback): classic COLMAP

```bash
colmap feature_extractor  --database_path db.db --image_path images
colmap sequential_matcher --database_path db.db
colmap mapper             --database_path db.db --image_path images --output_path sparse
colmap model_converter    --input_path sparse/0 --output_path points.ply --output_type PLY
```

- Installed on the Mac (`brew install colmap`) → gives an end-to-end run **without the GPU machine**;
  Linux distribution packages ship without CUDA, a CUDA build is separate work.
- Produces the same sparse model format, so it adds no integration code of ours.
- Natively provides the reprojection error and the number of registered images.

## 5. What the contract does not promise

- **Resuming from the middle** of the heavy stage: in variant A it is a single forward pass, there
  is no state. A restart is only from scratch.
- Metric scale: both variants give a result with undetermined scale →
  `scale_status = not_determined`.
- Version compatibility: no installation has been performed. Versions are fixed at the first
  actual installation and recorded in the manifest.
