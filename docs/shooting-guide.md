# Shooting guide

This document ships with v1. It describes **how to shoot a video suitable for
reconstruction**, and what the system does not expect from you.

> **Important about the status of these recommendations.** The numbers below (duration, number of loops, distance)
> are a starting point chosen on general grounds, **not** measured thresholds. The first
> end-to-end run (M1) will show what works in practice, and the document will be refined. Where
> a value has already been confirmed by measurement, this will be stated explicitly.

---

## 1. What the system checks and what it takes on trust

The distinction is essential: if a condition from the second column is violated, the tool
**will not find out** and will not warn you — it will produce a bad result or none at all.

| Checked automatically | Taken on trust from you |
|---|---|
| the file is readable, the container and codec are supported | the object is static |
| duration, resolution, frame count, frame rate | the object is rigid and opaque |
| frame orientation (rotation is applied) | the object has enough texture |
| frame sharpness (numerically) | the lighting is stable |
| difference between adjacent frames (numerically) | viewpoints overlap, the walk-around is complete |

Warnings such as "the video has many blurry frames" are **observations based on numbers**, not
an established cause. The system cannot distinguish camera shake from object movement.

---

## 2. The object

**Suitable:**

- a single object, static throughout the shoot;
- rigid — does not crumple, bend or sag;
- opaque and non-glossy;
- with pronounced texture: a pattern, lettering, roughness, uneven color.

**Not suitable (v1 does not support this):**

- transparent, mirror-like, strongly specular (glass, chrome, polished metal);
- plain smooth single-color (a white mug, a sheet of paper, a matte ball);
- soft and shape-changing (fabric, a pillow, a living creature);
- moving, or composite with moving parts.

**In practice:** if the object is smooth and single-color, the shoot will almost certainly not produce a result.
Pick a different object — it is cheaper than figuring out a failure.

---

## 3. Scene and lighting

- Place the object on a **textured surface**: a wooden table, a carpet, a newspaper.
  A plain white table is bad: the algorithm has nothing to latch onto around the object.
- The background is **not removed** and will remain in the result. This is normal for v1. Keep the background static:
  do not shoot by a window with cars passing, do not walk in front of the camera.
- The light is even and **constant**: diffuse daylight or general room lighting.
- Do not switch the light on or off mid-shoot, do not block the source with yourself.
- **Your shadows** move with the camera and interfere. Position yourself so as not to cast
  a shadow on the object.
- Do not use a flash or a flashlight: lighting that moves with the camera changes the appearance of the
  surface from frame to frame.

---

## 4. Shooting

**Trajectory.** Walk around the object in a circle, keeping it in the center of the frame. At least one full
loop, preferably two or three at different heights:

1. a loop at the object's level;
2. a loop from above at roughly 30–45°;
3. if needed, a loop closer to the surface.

**Speed.** Slowly. Guideline: a full loop in roughly 20–30 seconds. Fast movement
gives blurred frames, they will be rejected, and few useful viewpoints will remain.

**Duration.** 30–60 seconds is usually enough. Longer does no harm, but is not necessary:
the tool selects a limited number of frames anyway.

**Distance.** The object takes up a noticeable part of the frame, but fits into it entirely
with margin at the edges. Do not go up close: when shooting up close, adjacent viewpoints stop
overlapping.

**Overlap.** Adjacent frames must show a **common part** of the object. This is the main
condition: reconstruction relies on the same detail being visible from different sides.

**How to move:**

- hold the camera with both hands, elbows pressed to your body;
- walk smoothly, do not shift your feet in jerks;
- do not rotate the camera in place — you need to move **around** the object, not pivot
  on a single point;
- do not change the zoom while shooting.

---

## 5. Phone settings

- Format: ordinary video recording, **MP4 or MOV**, codec **H.264 or HEVC**. This is what the
  camera shoots by default.
- Resolution: 1080p is enough. 4K is acceptable, but the file will be heavy with no gain —
  frames are downscaled during processing anyway.
- Frame rate: 30 frames/s. A high frame rate (60, 120) is not needed.
- **Lock focus and exposure** if the phone allows (long-press on the object —
  AE/AF Lock). Automatic adjustment on the move changes brightness and sharpness, which looks like a change
  of the scene.
- Stabilization can be left on.
- Portrait modes, filters, HDR effects and "cinematic" mode — **turn off**:
  they alter the image nonlinearly.

---

## 6. What not to do

- Do not move the object during the shoot, even slightly.
- Do not shoot from one point while rotating the camera.
- Do not change the zoom or switch lenses mid-recording.
- Do not shoot against the light and do not allow overexposed highlights.
- Do not let strangers or objects pass through the frame.
- Do not stitch several videos into one: one run — one continuous video.

---

## 7. Check before processing

```bash
v3d check <your_video.mov>
```

The command will tell you whether the file is supported and show its properties. It does **not assess** the quality
of the shoot — only the format.

```bash
v3d prepare <your_video.mov> --budget 80
```

Here the observations appear. What to look at:

| What you see | What it means | What to do |
|---|---|---|
| `frames selected` noticeably below the budget | few usable frames | shoot slower and longer |
| warning `many_blurry_frames` | many blurry frames | shoot more smoothly, with more light |
| warning `low_viewpoint_diversity` | adjacent frames differ only slightly | move around the object more actively |
| warning `short_video` | the video is short | shoot longer |
| error "usable frames fewer than the minimum" | preparation did not take place | reshoot |

A warning **does not block** processing. It says that the result may fail —
and does not name a cause, because the system does not know it.

---

## 8. Short checklist

- [ ] the object is rigid, opaque, textured
- [ ] the object stands on a textured surface and will not move
- [ ] the light is even, does not change, the flash is off
- [ ] focus and exposure are locked, effects are off
- [ ] the walk-around is complete, 1–3 loops, slowly, 30–60 seconds
- [ ] the whole object is in the frame throughout
- [ ] there are no extraneous moving objects in the frame
- [ ] after shooting: `v3d check`, then `v3d prepare`, read the warnings

---

## 9. If the result did not turn out

This is a normal outcome, especially in the first attempts. The order of analysis:

1. Read the run's `report.md`: how many frames were selected, how many cameras were registered.
2. Few registered cameras — almost always a shooting problem: little texture, little
   viewpoint overlap or blur.
3. Swap the object for one that is clearly textured and repeat. If it worked on that —
   the problem was the object, not the tool.
4. Record the failure: in this project a negative result is kept on a par with
   a successful one, and not erased.
