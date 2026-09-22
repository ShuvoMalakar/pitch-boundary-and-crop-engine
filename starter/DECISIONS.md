# Engineering Decisions & Rationale

Key assumptions, trade-offs, and design choices made while building the pitch boundary and crop engine.

---

## 1. Assumptions & Open Questions

### Assumptions
- **Camera Motion & Sampling Rate:** Broadcast cameras track gameplay with smooth panning over 1–3 seconds, so adjacent frames share high visual redundancy. Sampling at 3 FPS captures boundary changes reliably while eliminating ~90% of frame decoding and compute overhead. If a downstream consumer later needs to render a continuous cropped video stream rather than layout metadata, we can sample denser or interpolate between keyframes.
- **Missing Boundaries as Expected States:** Real matches frequently cut to player close-ups, bench shots, referee reactions, or blackouts. Missing boundary lines are normal operational states, not system crashes. The engine flags them, records them in telemetry, and keeps running.
- **Pluggable Detector Boundary:** In production, color thresholding will be replaced with a deep learning model (e.g. YOLO, SAM). I defined a single interface (`FieldDetector`) so downstream geometry, metrics, and orchestration remain completely agnostic of the underlying model.

### Questions for Product & ML
1. **Model Latency & Hardware:** What is the target inference latency for the production model? If it exceeds ~50ms per frame, we should move from `ThreadPoolExecutor` to multi-processing or GPU batching to avoid GIL contention.
2. **Crop Consumption:** How will the crop coordinates be consumed? Does the downstream service expect discrete keyframe updates, or smoothed transitions for live broadcast?
3. **Deployment Topology:** Will multi-camera matches run as separate container workers per feed reporting to a central orchestrator, or should a single worker process multiple live streams concurrently?

---

## 2. Validation Strictness vs. Fallback

### Where I Fail Fast
- **Configuration & CLI Inputs:** Bad inputs should fail before burning compute on a video feed. I used Pydantic v2 with `extra="forbid"` across all config models. Unknown keys, invalid aspect ratios, negative padding, or out-of-range thresholds cause immediate startup termination with clear errors.
- **Unreadable Video:** If OpenCV cannot open or decode the source video, the engine raises `VideoSourceError` immediately rather than silently producing empty results.

### Where I Allow Fallbacks & Fault Isolation
- **Noisy Contours:** Minor contour self-intersections from segmentation masks are repaired via `poly.buffer(0)` before discarding them.
- **Missing Detections:** Frames without clear pitch boundaries are marked invalid and skipped without halting the pipeline.
- **Network Fault Isolation:** Network failures talking to `mock_api` are logged as warnings and do not crash or abort video processing. An external reporting outage should never kill an active analysis job.

---

## 3. Performance Trade-offs

1. **Sub-Linear Frame Seeking (3 FPS):** Using `cap.set(cv2.CAP_PROP_POS_FRAMES, ...)` to sample 3 FPS reduces a 90-minute match from 162,000 frames to 16,200. This delivers a 10x compute/IO reduction, directly satisfying the requirement that processing scales with needed frames.
2. **Invariant Geometry Caching:** The 1280x720 frame boundary polygon is constructed once at initialization in `GeometryCalculator`, eliminating thousands of redundant object allocations.
3. **Multi-Threading over Multi-Processing:** OpenCV and Shapely release Python's GIL during their C/C++ operations. `ThreadPoolExecutor` provides multi-core concurrency without the IPC serialization and memory overhead of separate processes.
4. **Median Spatial Aggregation:** Deriving the final crop from the median area detection filters out transient camera cuts and blurs, avoiding the distorted shapes that direct vertex averaging would cause.

---

## 4. AI / LLM Disclosure

I used an AI assistant (Antigravity) as an interactive pair-programmer:

- **What was prompted for:** Scaffolding initial boilerplate (Pydantic model skeletons, test signatures, Dockerfile base layers) and looking up OpenCV contour hierarchy flags.
- **What was designed and reviewed by hand:** The core architecture (the `FieldDetector` seam, library vs CLI separation, domain exception hierarchy, wire models), performance optimizations (sub-linear frame seeking, geometry caching, concurrency model), network fault isolation, dependency hygiene, and all code reviews.
