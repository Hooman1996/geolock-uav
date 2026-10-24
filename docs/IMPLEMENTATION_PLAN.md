# GeoLock Implementation Plan (v2)

This is the build plan for GeoLock. It turns the research spec (`GeoLock_Research_Spec.md`, v2, 17 Oct 2026) into ordered, small coding steps. The spec says *what* and *why*. This file says *in which order, in which file, and how we know a step is done*.

| | |
|---|---|
| Version | v2, 24 Oct 2026. Changes from v1: work on `main` until the first working version; datasets are downloaded by hand (list in Step 00); Python 3.12; every matcher and dataset uses its official code (checked in §4). |
| Start of coding | Mon 26 Oct 2026 |
| Target | BMVC 2027 (deadline expected around late May 2027), with a decision point in mid-Feb 2027 (spec §10.1) |
| Source of truth for maths | Research spec §5 (section numbers below, like "§5.4", point there) |

---

## 1. Rules that keep the code simple

These rules apply to every step. When in doubt, pick the simpler option.

1. **Plain functions on numpy arrays.** Geometry code is a set of small pure functions (`array in, array out`). No class hierarchies. The only abstract class in the whole project is `Matcher`.
2. **One flat package.** About 12 modules in `geolock/`, each under ~300 lines. No sub-sub-packages.
3. **One pipeline, many settings.** Every ablation condition (1 to 8) is a `Condition` value passed to the same `run_frame()` function. Never copy the pipeline to make a variant.
4. **Configs stay small.** Conditions are defined once in Python (`conditions.py`). A YAML file per experiment only lists *which* dataset, matchers, conditions, search radii and seeds to run. No Hydra, no config inheritance.
5. **Results are flat tables.** One row per (frame, condition, matcher, seed), saved as Parquet. All metrics, plots and statistics are computed later from these tables, so we never re-run a matcher to get a new number.
6. **Every step ends with a passing test.** Geometry is tested on synthetic scenes with known answers before any real data is used.
7. **Build things only when an experiment needs them.** The DEM orthorectification (6b), PnP (O4), feature caching, uncertainty and Jetson work are all scheduled late, right before the experiment that uses them.
8. **Official code for every matcher and dataset.** Each matcher runs from its authors' repository, pinned to a commit (§4). We write one short adapter file per matcher and no wrapper library sits in between. Standard libraries do the rest: OpenCV for warps and RANSAC, pyproj for coordinates, rasterio for GeoTIFFs, geographiclib for distances.
9. **One metric map frame.** The pipeline works in metres in each map's own frame (UTM for UAV-VisLoc, the scene-local frame for AnyVisLoc, the game plane for GTA-UAV). Latitude and longitude are produced only where the map is geo-referenced.
10. **The pipeline never reads ground truth.** A test enforces it (Step 18).

### Explicitly out of scope (unless an experiment later proves we need it)

ROS or real-time integration, C++ code, our own CUDA kernels, training or fine-tuning matchers, a sequential filter (EKF), cross-modal matching, a web UI, distributed job systems, matcher wrapper libraries, and Cloud-Optimised GeoTIFF tooling (plain GeoTIFF per level is enough).

---

## 2. How we use git

- **Steps 01 to 17 are committed directly to `main`.** These early steps are small and nothing works end to end yet, so branches would only add overhead.
- **At the end of Step 17** (the first real localization works), we tag `v0.1`. That is the first working version.
- **From Step 18 on, each step gets its own branch and pull request** (`step-18-conditions`, `step-19-learned-matchers`, ...). `main` then always holds a working pipeline. A step is merged when its "Done when" test passes in CI.
- Large data, weights and results never go into git (`data/`, `weights/` and `results/` are in `.gitignore`).

---

## 3. Repository layout (target)

This is a simplified version of spec §5.10. Files appear as the steps below create them.

```
geolock-uav/
├── geolock/
│   ├── types.py          # Frame, Condition, MatchResult, FrameResult dataclasses
│   ├── camera.py         # K, distortion, undistort points, f_px
│   ├── rotations.py      # Euler -> R, NED<->ENU, gimbal chain, tilt, heading, nadir pixel
│   ├── geo.py            # map frame <-> pixels, UTM<->WGS84, distance, AGL, DEM lookup
│   ├── scalelock.py      # GSD formulas + level / working-resolution choice (C1)
│   ├── warp.py           # resize, yaw homography, virtual-nadir rectification (+ DEM ortho later)
│   ├── robust.py         # MAGSAC++ homography/similarity fit + sanity checks
│   ├── observe.py        # O1 centre, O2 nadir, O3 pose-from-H, O4 PnP-DEM
│   ├── pyramid.py        # build and read the multi-level reference map
│   ├── search.py         # seeded priors, search window, tiling (§5.8)
│   ├── conditions.py     # the 8 ablation conditions as Condition values
│   ├── pipeline.py       # run_frame(frame, condition, matcher, pyramid) -> FrameResult
│   ├── matchers/
│   │   ├── base.py       # Matcher ABC + registry
│   │   ├── opencv.py     # SIFT, AKAZE
│   │   ├── lightglue.py  # SuperPoint+LG, ALIKED+LG (cvg/LightGlue)
│   │   ├── roma.py       # RoMa v1 (Parskatt/RoMa), RoMa v2 (Parskatt/RoMaV2)
│   │   ├── eloftr.py     # EfficientLoFTR (zju3dv/EfficientLoFTR)
│   │   └── ...           # one file per extra matcher, added in Step 26
│   ├── datasets/
│   │   ├── gta_uav.py
│   │   ├── anyvisloc.py
│   │   └── uav_visloc.py
│   └── eval/
│       ├── metrics.py    # A@T, median, RMSE, success, AUC
│       ├── stats.py      # bootstrap CI by flight, paired tests, mixed model
│       └── plots.py
├── third_party/          # git submodules of official repos that are not pip packages (§4.3)
├── configs/
│   ├── datasets/*.yaml   # paths, frame/CRS, altitude datum, camera, licence notes
│   └── experiments/*.yaml
├── scripts/
│   ├── build_pyramid.py
│   ├── run.py            # run one experiment YAML -> results/*.parquet
│   └── report.py         # results -> tables and figures
├── tests/
├── docs/
├── pyproject.toml
└── README.md
```

Data lives outside git, in `data/` (ignored): `data/raw/`, `data/pyramids/`, `data/results/`. Matcher weights go to `weights/` (ignored) or the torch hub cache.

### The four core types (`geolock/types.py`)

```python
@dataclass
class Frame:
    frame_id: str
    image: np.ndarray            # HxWx3 uint8, native resolution
    K: np.ndarray                # 3x3 intrinsics
    dist: np.ndarray | None      # distortion coefficients, or None
    R_wc: np.ndarray | None      # 3x3 camera -> map-frame rotation (E, N, Up), None if attitude unknown
    h_agl: float | None          # metres above ground (after the datum step, §5.2)
    prior_EN: tuple[float, float]  # search prior from the protocol (§5.8), map frame metres
    gt_EN: tuple[float, float]     # ground truth, for scoring only
    flight_id: str               # used for bootstrap by flight

@dataclass(frozen=True)
class Condition:
    name: str                                        # "1", "2", ..., "6b", "7", "8"
    scale: Literal["fixed", "nadir", "tilt"]
    scalelock: Literal["SL-R", "SL-D"] = "SL-R"
    yaw: Literal["none", "inplane", "exact"] = "none"
    tilt: Literal["none", "rectify_flat", "rectify_dem"] = "none"
    model: Literal["homography", "similarity"] = "homography"

@dataclass
class MatchResult:
    kpts_q: np.ndarray           # Nx2, query pixels (warped query)
    kpts_r: np.ndarray           # Nx2, reference pixels (level z*)
    conf: np.ndarray             # N

@dataclass
class FrameResult:               # one Parquet row
    frame_id: str; condition: str; matcher: str; seed: int
    level: int; gsd: float; query_scale: float
    n_matches: int; n_inliers: int; ok: bool; fail_reason: str
    H: list[float]               # 9 values, query-original -> map pixels
    est_EN: dict[str, tuple[float, float]]   # {"O1": .., "O2": .., "O3": .., "O4": ..}
    err_m: dict[str, float]      # filled by the scorer, not by the pipeline
    timings_ms: dict[str, float]
```

The observation model is **not** part of `Condition`. O1 to O4 are all computed from the same fitted homography on every run, so they come for free and are compared offline (spec §6.2).

---

## 4. Official code: what we use and what we found

All repositories below were cloned and read on 24 Oct 2026. The commit is the one we pin unless a later fix is needed.

### 4.1 Matchers

| Matcher | Official repo (commit) | Install | Licence | Python 3.12 | Notes |
|---|---|---|---|---|---|
| SIFT, AKAZE | OpenCV (`opencv-python`) | pip | Apache-2.0 | ✅ | |
| SuperPoint + LightGlue | `cvg/LightGlue` (`eb42fee2d714`) | pip from git | LightGlue Apache-2.0; **SuperPoint weights and inference file are non-commercial, academic only** (confirmed in repo) | ✅ | SuperPoint inside LightGlue is the port the LightGlue authors ship. |
| ALIKED + LightGlue | `cvg/LightGlue` (ALIKED weights from `Shiaoming/ALIKED`) | pip from git | BSD-3 / Apache-2.0 | ✅ | The ALIKED repo itself needs compiled custom CUDA ops. LightGlue's ALIKED loads the same official weights without them, so we use that. |
| EfficientLoFTR | `zju3dv/EfficientLoFTR` (`07e9c1401e71`) | git submodule | **Changed to "Project Registration License" (Sep 2026).** Free for academic research, publications allowed, no registration. Any company or organisation project must register first. | ⚠️ official env is Python 3.8 / torch 2.0 | Model code only needs torch, einops, kornia, loguru, yacs. It uses `kornia.utils.grid`, which kornia 0.8.3 removed, so we pin `kornia>=0.7.3,<0.8.3`. Weights are on Google Drive (manual download). |
| RoMa v1 / Tiny RoMa | `Parskatt/RoMa` (`77f8d6880352`, package `romatch`) | pip from git | MIT | ✅ (needs torch ≥ 2.5.1) | Weights auto-download (RoMa + DINOv2). |
| RoMa v2 | `Parskatt/RoMaV2` (`95c9968145c8`, package `romav2`) | pip from git | MIT code; DINOv3 under Meta's DINOv3 licence | ✅ officially tested on 3.12 | The spec called it `Parskatt/romav2`; the real name is `RoMaV2`. Weights auto-download from GitHub releases; the DINOv3 model code is fetched by `torch.hub` from GitHub on first run. |
| XFeat | `verlab/accelerated_features` (`e92685f57f83`) | git submodule (or `torch.hub`) | Apache-2.0 | ⚠️ README env is Python 3.8 | Core model is plain torch. Requirements pin old versions for demos only. Weights are in the repo. |
| Steerers (DeDoDe + steerer) | `georg-bn/rotation-steerers` (`b2becd5d99bc`) + `Parskatt/DeDoDe` (`6d156183f4dc`) | pip from git | MIT (DeDoDe-G uses DINOv2, Apache-2.0) | ✅ (needs ≥ 3.9) | Weights from GitHub releases. |
| GIM (LightGlue) | `xuelunshen/gim` (`f09105a0555e`) | git submodule | MIT | ⚠️ official env is Python 3.9 / torch 1.12 | Its LightGlue code is pure torch + omegaconf, so it should load in our env. `gim_lightglue_100h.ckpt` is in the repo; GIM-RoMa weights are on Google Drive. |
| MASt3R | `naver/mast3r` (`f5209afc300c`, plus `dust3r` submodule) | git submodule | **CC BY-NC-SA 4.0** (code and checkpoints) | ⚠️ tested on 3.11 | Optional `curope` CUDA kernel compile. Added late (Step 26) because it is the heaviest install. |
| MatchAnything | `zju3dv/MatchAnything` (`8cd8c1129a6d`) | — | Project Registration License | ❌ | **The official GitHub repo contains no code.** Code and weights exist only inside the authors' Hugging Face Space. We drop it and use GIM as the generalist matcher. |
| LiteSAM | not found | — | — | — | Could not confirm public code. |
| DALGlue | not found | — | — | — | Could not confirm public code. |

### 4.2 Datasets

| Dataset | Official repo (commit) | What we found | Effect on the plan |
|---|---|---|---|
| AnyVisLoc / UAV-AVL | `UAV-AVL/Benchmark` (`6c8f7980f35d`) | Full release is 24 scenes, 20,077 images. Each scene has one JSON, aerial and satellite PNG maps with `.npy` DSMs, and one `.npz` per image (image, K, 5 distortion coefficients, `pose_c2w`, `xyz`, `euler_deg`). **Coordinates are scene-local metres, not UTM.** Official code is tested on Python 3.10, PyTorch 2.5.1, numpy < 2. Licence: research only, no commercial use (including model deployment and contract research), no redistribution of derived splits, poses or metadata. | Our loader reads the `.npz` and JSON files directly. Results are scored in the scene-local frame. We run their official baseline (for E6) in its own Python 3.10 environment. |
| GTA-UAV / Game4Loc | `Yux1angJi/GTA-UAV` (`1fbc3dbe452d`) | LR 512×384 is 12.8 GB, HR 1920×1440 is 143.3 GB, both on Hugging Face. Angle bug fixed 20 Oct 2025. Satellite references are 256-px tiles at 4 zoom levels, game-plane coordinates. `drone_loc_x_y` is the image-centre location; `height` is AGL. Apache-2.0. | The dataset is already a ×2 pyramid, so the pyramid builder just stitches tiles per level. |
| UAV-VisLoc | `IntelliSensing/UAV-VisLoc` (`92dab485bfc7`) | 16.4 GB full set and a 2.04 GB "minimal coordinate error" sample, on Google Drive or Baidu only. The README defines the label as the ground point below the drone, i.e. the image centre. **No licence file.** | Start with the 2 GB sample. Ask the authors about licence terms before publishing any derived material. |

### 4.3 How official code is installed

- **Pip-installable repos** (LightGlue, RoMa, RoMa v2, DeDoDe, Steerers) go into `pyproject.toml` as optional extras pinned to a commit, for example `lightglue @ git+https://github.com/cvg/LightGlue@eb42fee2d714`.
- **Repos that are not packages** (EfficientLoFTR, XFeat, GIM, MASt3R) are git submodules in `third_party/`, pinned to a commit. Each adapter adds its submodule to `sys.path` in one line.
- **One environment** (Python 3.12, CUDA PyTorch ≥ 2.5.1) holds every matcher. If one official repo cannot run in it, we give that matcher its own environment rather than patching its code.

---

## 5. Phases and steps

Each step lists the files it touches, what goes in them, and a "Done when" test. Steps in the same phase can overlap, but each phase depends on the one before.

### Phase 0: Setup (week of 26 Oct)

#### Step 00. Download datasets (by hand)

hooman downloads these. Coding does not wait for them: Phase 1 needs no real data, and Step 13 needs only GTA-UAV.

| Priority | What | Where | Size | Needed by |
|---|---|---|---|---|
| 1 | GTA-UAV **LR** (for development) | huggingface.co/datasets/Yux1ang/GTA-UAV-LR | 12.8 GB | Step 13 |
| 2 | UAV-VisLoc **sample** | Google Drive link in the UAV-VisLoc README ("Dataset Example") | 2.04 GB | Step 15 |
| 3 | AnyVisLoc (full) | Official portal linked in the `UAV-AVL/Benchmark` README (OneDrive or Baidu). Accept the dataset licence. | large | Step 14 |
| 4 | Copernicus GLO-30 DEM tiles covering the 11 UAV-VisLoc sites | Copernicus DEM on AWS Open Data (`copernicus-dem-30m`) or OpenTopography | small | Step 15 |
| 5 | GTA-UAV **HR** (for experiments) | huggingface.co/datasets/Yux1ang/GTA-UAV-HR | 143.3 GB | Step 23 |
| 6 | UAV-VisLoc full set | Google Drive link in the README ("Overall Dataset") | 16.4 GB | Step 23 |
| 7 | EfficientLoFTR outdoor weights (`eloftr_outdoor.ckpt`) | Google Drive link in the EfficientLoFTR README | small | Step 19 |

Other weights (LightGlue, RoMa, RoMa v2, XFeat, GIM-LightGlue, Steerers) download automatically or come with the repo.

- **Done when:** items 1 to 4 are on disk, and `configs/datasets/*.yaml` records each local path.

#### Step 01. Repository skeleton
- `pyproject.toml`: **Python 3.12** (`requires-python = ">=3.12,<3.13"`). Core deps: numpy, opencv-python, scipy, pyproj, rasterio, geographiclib, pandas, pyarrow, pyyaml, pytest. Optional extra `[gpu]`: torch ≥ 2.5.1, kornia ≥ 0.7.3 < 0.8.3, einops, loguru, yacs, omegaconf, plus the pinned official repos from §4.3.
- `ruff` for lint and format, `pytest` for tests, a GitHub Actions workflow that runs both on CPU with Python 3.12.
- `README.md` with install and "run the tests" instructions. `.gitignore` covering `data/`, `weights/`, `results/`.
- **Done when:** `pip install -e .` and `pytest` pass in CI on an empty test.

#### Step 02. Core types
- `geolock/types.py` with the four dataclasses above.
- **Done when:** a test builds each type and round-trips a `FrameResult` through a Parquet file.

### Phase 1: Geometry core, no real data needed (26 Oct to 22 Nov)

All of Phase 1 is pure numpy and OpenCV, tested on synthetic inputs. This is where most correctness bugs would hide, so it is done first and tested hardest.

#### Step 03. Camera model
- `geolock/camera.py`: `K_from_fov(fov_deg, W, H)`, `K_from_sensor(f_mm, sensor_w_mm, W, H)`, `f_px(K)`, `undistort_points(pts, K, dist)`, `scale_K(K, s)`.
- **Done when:** FOV ↔ focal round-trip tests pass, and the DJI Mavic 3 example from spec §5.3.2 gives 0.0267 m/px at 100 m.

#### Step 04. Rotations and attitude
- `geolock/rotations.py`: `R_ned_from_body(roll, pitch, yaw)` (ZYX), `R_ENU_FROM_NED`, `R_GIMBAL_FROM_CAM`, `R_wc_from_attitude(...)` for strapdown and gimbal, plus `tilt(R_wc)`, `heading(R_wc)`, `nadir_pixel(K, R_wc)`.
- **Done when:** the spec §5.12 formula tests pass: gimbal pitch −90° points the optical axis straight down; strapdown tilt satisfies cos t = cos φ · cos θ; the nadir pixel equals the principal point for a nadir camera.

#### Step 05. Coordinates, distances and AGL
- `geolock/geo.py`: `pixel_to_EN(geotransform, u, v)` and inverse (GDAL pixel-centre +0.5 convention), `distance_m(EN_a, EN_b)`, and for geo-referenced maps only `utm_epsg_for(lat, lon)`, `to_utm(lat, lon)`, `to_wgs84(E, N)` (pyproj) and `geodesic_m(latlon_a, latlon_b)` (Karney). Also `agl_from(source, value, ...)` for the cases in spec §5.2, and `dem_sample(dem, E, N)`.
- **Done when:** pixel → UTM → WGS84 → UTM → pixel round-trip is within 1e-6 px, and the AGL resolver reproduces three hand-worked cases (ellipsoidal, MSL, take-off relative).

#### Step 06. GSD formulas and ScaleLock (C1)
- `geolock/scalelock.py`: `gsd_nadir(h, f_px)`, `gsd_tilt(h, f_px, t)` returning (g_perp, g_par, g_eff), and `select_level(h, f_px, W0, H0, tilt, budget_px, level_gsds, variant)` returning `(level, gsd, query_scale, footprint_m)`. This is the §5.4.2 algorithm in about 30 lines. Selection uses log-ratio distance.
- **Done when:** the numeric check in spec §5.3.3 passes (H = 100 m, f = 1000 px, t = 30° gives 0.1333 and 0.1155 m/px), g_eff matches a finite-difference g_iso at the principal point, and selection on a ×2 pyramid never leaves a residual above √2 for SL-D or above 1e-6 for SL-R.

#### Step 07. Query warps
- `geolock/warp.py`: one function per warp, each returning a 3×3 homography so they compose:
  - `H_resize(s)`
  - `H_yaw_inplane(K, psi)` (approximation, condition 3′/4′ only)
  - `H_yaw_exact(K, R_wc)` (rotation about world vertical, §5.5.3)
  - `H_rectify_flat(K, R_wc, h_agl, gsd, canvas)` (virtual nadir camera, §5.5.4)
  - `apply(image, H, out_size)` wrapping `cv2.warpPerspective` once. Warps are always composed first and applied once.
- **Done when:** a synthetic north-pointing road becomes vertical after `H_yaw_exact`, and the exact yaw reduces to the in-plane rotation for a nadir camera.

#### Step 08. Synthetic scene renderer (test fixture)
- `tests/synthetic.py`: take any orthophoto with a known geotransform, a camera (K, R_wc, C), and render the frame the drone would see over flat ground. Returns the frame and the true position.
- This is the most valuable test tool in the project: every later geometry step is checked against it.
- **Done when:** rendering a nadir frame and rectifying it with `H_rectify_flat` reproduces the orthophoto crop at g* (pixel error under 0.5 px).

#### Step 09. Robust fit and sanity checks
- `geolock/robust.py`: `fit(kpts_q, kpts_r, conf, model, threshold_px, seed)` using `cv2.findHomography(..., cv2.USAC_MAGSAC)` (or `estimateAffinePartial2D` for the similarity model), and `sanity_check(H, n_inliers, ratio, footprint, yaw_prior)` returning `(ok, reason)` with the five checks from spec §5.7.
- **Done when:** with synthetic matches plus 50% outliers it recovers H to within 0.5 px, and each sanity check rejects a hand-made bad case.

#### Step 10. Observation models O1, O2, O3
- `geolock/observe.py`: `o1_centre(H_tot, K, geo)`, `o2_nadir(H_tot, K, R_wc, geo)` (with the fallback for a nadir pixel far outside the image), `o3_pose_from_H(H_tot, K, geo, z_ground)` returning camera centre C plus recovered height and attitude.
- **Done when:** spec §5.12 test 1 passes on synthetic tilted frames (0°, 15°, 30°, 45°): O2 and O3 recover the drone position within 0.1·g, and O1 shows the expected bias H·tan t.

**Milestone M1 (about 22 Nov): geometry core done.** Every formula in spec §5.3 to §5.6 (except DEM ortho and PnP) is implemented and tested on synthetic scenes.

### Phase 2: Maps and data (16 Nov to 13 Dec)

#### Step 11. Reference pyramid builder
- `geolock/pyramid.py` + `scripts/build_pyramid.py`: two input paths, one output format.
  - From one map image (UAV-VisLoc TIFF reprojected to UTM; AnyVisLoc PNG in its scene-local frame): write levels g_k = g_0·2^k with area-average downsampling.
  - From an existing tile pyramid (GTA-UAV): stitch the 256-px tiles of each zoom level into one image per level.
  - Output: one plain GeoTIFF per level plus one `pyramid.json` (frame or CRS, per-level GSD, geotransform, size, source, licence, DEM path).
  - Reader: `Pyramid.load(path)`, `pyramid.crop(level, E, N, size_m)`.
- Optional flag `--step sqrt2` for the √2 ablation (E1).
- **Done when:** building a pyramid from a test GeoTIFF gives the right GSD per level, and a crop at any level lands on the same ground point (checked by geotransform).

#### Step 12. Search-window protocol
- `geolock/search.py`: `make_prior(gt_EN, R, seed, frame_id)` (uniform in a disc, deterministic per frame), `window(prior_EN, footprint_m, R)` (side S = F + 2R + 2m), `tiles(window, footprint_m, budget_px, gsd)` (tile side 1.5F, stride 0.5F).
- **Done when:** the same seed gives the same prior, the true position is always inside the window, and the tile count matches (S/F)² as expected.

#### Step 13. GTA-UAV loader + first metadata checks (E0, part 1)
- `geolock/datasets/gta_uav.py`: reads the official `*-drone2sate-*.json` files and yields `Frame` objects (exact AGL, attitude, game-plane coordinates). Start here because GTA-UAV has exact AGL, no distortion, and is public.
- `scripts/e0_fit_fov.py`: fit the unknown FOV from the dataset's IoU-paired tiles (spec §5.12 test 5) and confirm the angle naming (the README example has nadir in `cam_roll` ≈ −90°).
- **Done when:** the fitted FOV is recorded in `configs/datasets/gta_uav.yaml`, and predicted footprints overlap the dataset's paired tiles.

#### Step 14. AnyVisLoc loader (E0, part 2)
- `geolock/datasets/anyvisloc.py`: reads each scene's `Lxx_reference.json` and the per-image `.npz` (K, distortion, `pose_c2w` → R_wc, `xyz`), satellite and aerial map modes. AGL = z − DSM(nadir). All in the scene-local metric frame.
- **Done when:** `pose_c2w` and `euler_deg` agree after conversion on all samples (checked by script), and AGL values are positive and plausible.

#### Step 15. UAV-VisLoc loader (E0, part 3)
- `geolock/datasets/uav_visloc.py`: reproject site TIFFs to UTM, look up the DEM to turn `height` into AGL, FOV from EXIF or the Game4Loc assumption (52°×36°).
- `scripts/e0_visloc_datum.py`: decide per site whether `height` is MSL, using the O3 recovered height on a few frames (spec §8.2).
- **Done when:** the datum decision per site is written in `configs/datasets/uav_visloc.yaml`.

**Milestone M2 (about 13 Dec): data ready.** All three datasets load as `Frame` objects, each has a pyramid, and E0 open questions (FOV, datum, angle conventions) are answered.

### Phase 3: Matching and the pipeline (30 Nov to 27 Dec)

#### Step 16. Matcher interface + SIFT
- `geolock/matchers/base.py`: `Matcher` ABC with `name`, `detector_based`, `match(img_q, img_r) -> MatchResult`, and a tiny registry `get_matcher(name)`.
- `geolock/matchers/opencv.py`: SIFT + ratio test, AKAZE.
- **Done when:** SIFT on a synthetic frame against its orthophoto produces matches that `robust.fit` turns into the correct H.

#### Step 17. First end-to-end localization (Condition 1 and 2, SIFT, GTA-UAV)
- `geolock/pipeline.py`: `run_frame(frame, condition, matcher, pyramid)` doing, in order: choose level → build and apply query warp → crop window and tiles → match each tile → fit + sanity → compute O1..O3 → record timings. About 150 lines.
- `geolock/conditions.py`: conditions 1 and 2 first.
- **Done when:** on 100 GTA-UAV frames, condition 2 (ScaleLock) localizes clearly better than condition 1 with SIFT. *This is the first real result and the project's main go/no-go check.* Tag `v0.1`; from here on, one branch per step (§2).

#### Step 18. All conditions + leakage and determinism tests
- Add conditions 3, 4, 5, 6a, 7 to `conditions.py` (6b and 8 come in Step 22).
- Tests: setting `gt_EN` to NaN changes no output (spec §5.12 test 6); same seed gives an identical table (test 7).
- **Done when:** each condition runs on the synthetic scene and on 20 real frames without error, and both tests pass.

#### Step 19. Learned matchers from official repos
- Add the official repos for the three scoping matchers (§4.3) and one adapter file each:
  - `matchers/lightglue.py`: SuperPoint+LightGlue and ALIKED+LightGlue (`cvg/LightGlue`).
  - `matchers/roma.py`: RoMa v2 (`Parskatt/RoMaV2`), with RoMa v1 in the same file.
  - `matchers/eloftr.py`: EfficientLoFTR (`third_party/EfficientLoFTR`, sides divisible by 32).
- Each adapter is about 30 to 60 lines: convert our image to the tensor the official code expects, call it, return `MatchResult` in our pixel coordinates. The input size is set by ScaleLock, and dense matches are capped at 5,000 (RoMa's own `sample`).
- **Done when:** all of them run on the same 20 frames with the same pipeline call in the Python 3.12 environment, each passes a synthetic-pair test (known H recovered), and the input size passed to each model is logged and correct.

#### Step 20. Experiment runner and results table
- `scripts/run.py experiments/<name>.yaml`: loops dataset × matcher × condition × seed, writes one Parquet file per (matcher, condition), skips work already done (resume), and logs per-stage timings (`torch.cuda.synchronize` around GPU stages).
- Scoring is a separate function that adds `err_m` from `gt_EN`, so the pipeline itself never sees ground truth.
- **Done when:** a small YAML (2 conditions × 1 matcher × 50 frames) runs, is interrupted, resumes, and produces the same table as an uninterrupted run.

#### Step 21. Metrics and report
- `geolock/eval/metrics.py`: A@T (1, 3, 5, 10, 20, 50, 100 m), median / mean / RMSE (all frames with cap, and successful frames), success rate, AUC@100 m, RMSE@30, matcher calls per frame.
- `geolock/eval/stats.py`: bootstrap CI resampling flights, Wilcoxon on log error, McNemar on success, Holm correction.
- `scripts/report.py`: results → Markdown/LaTeX tables + CDF plots, with breakdowns by altitude and tilt bin.
- **Done when:** the report reproduces hand-computed metrics on a toy table.

**Milestone M3 (about 27 Dec): pipeline complete.** Any (dataset, matcher, condition) combination runs from one YAML, and one command makes the tables.

### Phase 4: Experiments (Jan to Feb 2027)

Each experiment is one YAML in `configs/experiments/` plus, if needed, the small feature it requires.

#### Step 22. Terrain-aware extras: 6b DEM orthorectification + O4 PnP
- In `warp.py`: `ortho_dem(frame, dem, prior_EN, gsd)` (inverse mapping, optional 2nd pass at the first estimate).
- In `observe.py`: `o4_pnp_dem(kpts_q, kpts_r, geo, dem, K, dist)` with `cv2.solvePnPRansac` (SQPnP, then LM).
- Add conditions 6b and 8.
- **Done when:** on a synthetic scene with relief, O4 recovers the position within 0.1·g and 6b beats 6a.

#### Step 23. E1: ScaleLock vs. baselines (C1)
- Baselines in `scalelock.py`: fixed level, brute force max-inlier, brute force oracle, query-resample-to-finest, SL-D, SL-R, SL-R ±1, √2 pyramid.
- Run on GTA-UAV (HR), UAV-VisLoc and AnyVisLoc-satellite, 3 matchers, R = 100 m, 3 seeds.
- **Output:** accuracy vs. matcher calls per frame, fraction of frames where the chosen level is not the finest.

#### Step 24. E2 + E2b: attitude factorial and sensor noise (C1b, C1c)
- Run conditions 1 to 8 on AnyVisLoc (both map modes) and GTA-UAV, stratified by tilt bin.
- Add a `perturb()` function that changes only the front-end inputs: roll/pitch, yaw, altitude scale, datum mistake, camera–IMU bias (spec §6.3).
- Add the mixed model to `stats.py` (statsmodels MixedLM on log error).
- **Output:** per-tilt-bin tables, interaction estimates, degradation curves.

#### Step 25. Uncertainty and integrity flags
- `geolock/uncertainty.py`: covariance from the homography fit (spec §5.9), NEES and coverage; integrity flag from O3 height and attitude vs. the sensors.
- **Output:** calibration table, failure-detection precision and recall.

#### Step 26. E3: full matcher benchmark (C2)
- Add the remaining official matchers one at a time, one adapter file each: Tiny RoMa, XFeat, Steerers, GIM-LightGlue, MASt3R (last, heaviest install). Each also gets a config entry with its input budget and licence.
- Run under the best E1+E2 configuration and under vanilla.
- **Output:** main results table and accuracy-vs-latency Pareto plot.

#### Step 27. E4 + E9: search radius and rotation sweeps (cheap)
- E4: R ∈ {50, 100, 250, 500} m. E9: synthetic in-plane rotation 0 to 180° on GTA-UAV nadir frames.
- **Output:** accuracy vs. R; per-matcher rotation curves.

#### Step 28. E5: reference feature caching (C3)
- `geolock/cache.py`: extract features on a fixed tile grid per level and save keypoints + fp16 descriptors (one `.npz` per tile is enough; move to HDF5 only if file counts become a problem).
- **Output:** MB per km² and ms per frame, with and without cache.

#### Step 29. E6: UAV-specific baselines
- Run the official AnyVisLoc baseline (`UAV-AVL/Benchmark`) in its own Python 3.10 environment, on the released 20k version.
- LiteSAM and DALGlue only if their code and weights become public. Otherwise reproduce the LiteSAM protocol on UAV-VisLoc and compare with published numbers, clearly labelled.

#### Step 30. Optional: E7 Jetson and E8 temporal gap
- Only if time allows before the February decision point.

**Milestone M4 (about 15 Feb 2027): decision point.** Choose BMVC 2027 vs. RA-L + IROS (spec §10.1).

### Phase 5: Paper and release (mid-Feb to May 2027)

#### Step 31. Figures, tables and paper draft
- All figures come from `scripts/report.py`, so every number in the paper can be regenerated from the Parquet results.

#### Step 32. Open release
- Code, configs, seeds, split generators and per-frame error tables (no dataset metadata, per the AnyVisLoc licence, spec §8.3). Tag a release.

**Milestone M5 (late May 2027): BMVC 2027 submission.**

---

## 6. Step titles at a glance

| # | Step | Phase | Branch | Milestone |
|---|---|---|---|---|
| 00 | Download datasets (by hand) | 0 Setup | — | |
| 01 | Repository skeleton | 0 Setup | main | |
| 02 | Core types | 0 Setup | main | |
| 03 | Camera model | 1 Geometry | main | |
| 04 | Rotations and attitude | 1 Geometry | main | |
| 05 | Coordinates, distances and AGL | 1 Geometry | main | |
| 06 | GSD formulas and ScaleLock | 1 Geometry | main | |
| 07 | Query warps | 1 Geometry | main | |
| 08 | Synthetic scene renderer | 1 Geometry | main | |
| 09 | Robust fit and sanity checks | 1 Geometry | main | |
| 10 | Observation models O1, O2, O3 | 1 Geometry | main | M1 Geometry core |
| 11 | Reference pyramid builder | 2 Data | main | |
| 12 | Search-window protocol | 2 Data | main | |
| 13 | GTA-UAV loader + FOV fit | 2 Data | main | |
| 14 | AnyVisLoc loader | 2 Data | main | |
| 15 | UAV-VisLoc loader + datum check | 2 Data | main | M2 Data ready |
| 16 | Matcher interface + SIFT | 3 Pipeline | main | |
| 17 | First end-to-end localization | 3 Pipeline | main | Go/no-go, tag `v0.1` |
| 18 | All conditions + leakage tests | 3 Pipeline | branch + PR | |
| 19 | Learned matchers from official repos | 3 Pipeline | branch + PR | |
| 20 | Experiment runner and results table | 3 Pipeline | branch + PR | |
| 21 | Metrics and report | 3 Pipeline | branch + PR | M3 Pipeline complete |
| 22 | 6b DEM ortho + O4 PnP | 4 Experiments | branch + PR | |
| 23 | E1 ScaleLock vs. baselines | 4 Experiments | branch + PR | |
| 24 | E2 + E2b attitude factorial + noise | 4 Experiments | branch + PR | |
| 25 | Uncertainty and integrity flags | 4 Experiments | branch + PR | |
| 26 | E3 full matcher benchmark | 4 Experiments | branch + PR | |
| 27 | E4 + E9 radius and rotation sweeps | 4 Experiments | branch + PR | |
| 28 | E5 feature caching | 4 Experiments | branch + PR | |
| 29 | E6 UAV-specific baselines | 4 Experiments | branch + PR | |
| 30 | E7 Jetson / E8 temporal (optional) | 4 Experiments | branch + PR | M4 Decision point |
| 31 | Figures, tables and paper draft | 5 Paper | branch + PR | |
| 32 | Open release | 5 Paper | branch + PR | M5 Submission |

---

## 7. Where this plan differs from the spec

| Spec says | Plan does | Why |
|---|---|---|
| `geometry/` sub-package with 10 files (§5.10) | 7 flat modules (`camera`, `rotations`, `geo`, `scalelock`, `warp`, `robust`, `observe`) | Fewer imports, each file has one clear job |
| vismatch wraps the learned matchers (§5.11) | Official repos, one adapter file per matcher | hooman asked for official code; no wrapper resizing to override |
| Python 3.11 (§5.11) | Python 3.12 | Requested; every core library and RoMa v2 support it (§4.1) |
| MatchAnything as generalist (§7.1) | GIM-LightGlue only | MatchAnything's official repo has no code (§4.1) |
| Everything reprojected to UTM (§5.1.2) | Each map's own metric frame; UTM only for geo-referenced maps | AnyVisLoc is scene-local and GTA-UAV is a game plane (§4.2) |
| Cloud-Optimised GeoTIFF + per-level JSON | Plain GeoTIFF per level + one `pyramid.json` | Maps are small enough to read whole |
| HDF5/zarr feature store | `.npz` per tile, only at Step 28 | Only E5 needs it |
| `Condition.observation` field | O1 to O4 always computed, compared offline | One run gives all four |
| Conditions defined in `experiments/*.yaml` | Conditions in Python, YAML only selects them | Conditions are code-reviewed and typed |
| Uncertainty and DEM ortho in the front-end phase | Moved to Phase 4, right before the experiment that needs them | Get a working end-to-end result first |

The maths is unchanged. Every formula still follows spec §5.

---

## 8. Hardware and environment

- **Phases 0 to 2 and SIFT runs:** any laptop, CPU only, Python 3.12.
- **Learned matchers (Step 19 onward):** one CUDA GPU with at least 16 GB. Dense runs are about 100+ GPU-hours in total (spec §6.5), so start E1/E2 on the 3 scoping matchers and stratified subsets.
- **Disk:** about 200 GB for GTA-UAV HR, UAV-VisLoc and AnyVisLoc together, plus pyramids and results.
- **Second environment:** Python 3.10 only for the official AnyVisLoc baseline (Step 29).
- **Jetson Orin:** only for optional Step 30.

## 9. Risks to the schedule

| Risk | What we do |
|---|---|
| AnyVisLoc download is slow | Steps 13 and 17 use GTA-UAV, which is public, so coding is not blocked |
| Step 17 shows no gain from ScaleLock | Stop and investigate before building more; this is the go/no-go point |
| An official repo does not run under Python 3.12 (most likely: EfficientLoFTR, GIM, MASt3R) | Give that matcher its own environment rather than patching the authors' code |
| UAV-VisLoc has no licence | Use it for results only; ask the authors before releasing anything derived from it |
| GPU time runs short | Subsample frames by tilt and altitude bin (spec §6.5); cache every match result |
