"""
Utility for mapping visual results to entities.

Core functionality: given a 2D bbox, look up in the segmentation image the
entity or body name with the highest occupancy ratio within it. Used by the
evaluation pipeline (scripts/codelab_eval.py) to parse bboxes output by the
VLM into concrete entity names in MuJoCo (e.g. "steak_0", "knob_rear_left").

Historical origin: the library-function portion of
original codelab_debug/test_ids_2_entity_name/test_ids_mapping.py (the rest
was a one-off debugging script and was not migrated).
"""
import numpy as np
from RMMBench.utils.utils import extract_base_name

# Keywords of targets detected at body level (rather than entity level)
BODY_LEVEL_TARGETS = ["knob", "button", "handle"]


def get_entity_from_bbox(env, bbox, target_entity_name=None, cam_id=3, min_ratio=0):
    """
    Look up in the segmentation image, from a 2D bbox, the entity or body with
    the highest occupancy ratio within it.

    Args:
        env: MuJoCo simulation environment
        bbox: [y_min, x_min, y_max, x_max] (image coordinates)
        target_entity_name: optional, target name output by the VLM (no suffix, e.g. "steak")
        cam_id: camera ID, default 3 (head camera)
        min_ratio: minimum occupancy ratio threshold

    Returns:
        str or None: full name of the recognized entity (e.g. "steak_0") or body name (e.g. "knob_rear_left")
    """
    _, _, seg, _, _ = env.get_camera_parse(cam_id)
    h, w = seg.shape[:2]

    # 1. Parse and clamp bbox boundaries
    y_min, x_min, y_max, x_max = [int(v) for v in bbox]
    y_min, y_max = max(0, y_min), min(h, y_max)
    x_min, x_max = max(0, x_min), min(w, x_max)

    bbox_area = (y_max - y_min) * (x_max - x_min)
    if bbox_area == 0:
        print("[get_entity_from_bbox] Empty bbox")
        return None

    # 2. Get or build the geom_id -> entity / body mapping cache
    geom_id_to_entity = getattr(env.task, '_geom_id_to_entity', None)
    geom_id_to_body = getattr(env.task, '_geom_id_to_body', None)
    if geom_id_to_entity is None or geom_id_to_body is None:
        geom_id_to_entity = {}
        geom_id_to_body = {}
        for entity_name, entity in env.task.entities.items():
            for geom in entity.geoms:
                geom_id = env.physics.bind(geom).element_id
                body_id = env.physics.bind(geom).bodyid
                body_name = env.physics.model.id2name(body_id, "body")
                geom_id_to_entity[geom_id] = entity_name
                geom_id_to_body[geom_id] = body_name
        env.task._geom_id_to_entity = geom_id_to_entity
        env.task._geom_id_to_body = geom_id_to_body
        print(f"[get_entity_from_bbox] Built geom_id_to_entity cache with {len(geom_id_to_entity)} entries")

    # 3. Count geom_ids within the bbox region
    bbox_seg = seg[y_min:y_max, x_min:x_max, 0]
    unique_geom_ids, counts = np.unique(bbox_seg, return_counts=True)

    # Determine whether body-level extraction is needed
    need_body_level = target_entity_name is not None and any(
        name in target_entity_name.lower() for name in BODY_LEVEL_TARGETS
    )

    if need_body_level:
        # 4a. Aggregate at body level (keep the entity prefix, e.g. stove_0/knob_front_right)
        body_counts = {}
        for geom_id, count in zip(unique_geom_ids, counts):
            if geom_id == 0:
                continue
            body_name = geom_id_to_body.get(geom_id)
            if body_name is None:
                continue
            # Keep the full body name (including the entity prefix) to distinguish same-named bodies under different entities
            body_counts[body_name] = body_counts.get(body_name, 0) + count

        if not body_counts:
            print("[get_entity_from_bbox] No bodies found in bbox")
            return None

        # 5a. Compute occupancy ratios and filter
        candidates = []
        for full_body_name, pixel_count in body_counts.items():
            ratio = pixel_count / bbox_area
            # Extract the pure body name for matching
            pure_body_name = full_body_name.split("/")[-1] if "/" in full_body_name else full_body_name
            # Body level: the name-match rule is that target_entity_name is a substring of pure_body_name
            name_match = (target_entity_name is None or target_entity_name.lower() in pure_body_name.lower())
            candidates.append({
                "body_name": full_body_name,
                "pixel_count": pixel_count,
                "ratio": ratio,
                "name_match": name_match,
            })
            print(f"  {body_name}: {pixel_count} pixels, ratio={ratio:.3f}, name_match={name_match}")

        # 6a. Prefer name matches, then sort by occupancy ratio
        matched = [c for c in candidates if c["name_match"]]
        if matched:
            best = max(matched, key=lambda x: x["ratio"])
        else:
            best = max(candidates, key=lambda x: x["ratio"])

        # 7a. Threshold check
        if best["ratio"] < min_ratio:
            print(f"[get_entity_from_bbox] Best ratio {best['ratio']:.3f} below threshold {min_ratio}")
            return None

        print(f"[get_entity_from_bbox] Selected body: {best['body_name']} (ratio={best['ratio']:.3f})")
        return best["body_name"]

    else:
        # 4b. Aggregate at entity level (original logic)
        entity_counts = {}
        for geom_id, count in zip(unique_geom_ids, counts):
            if geom_id == 0:  # Exclude background
                continue
            entity_name = geom_id_to_entity.get(geom_id)
            if entity_name is None:
                continue
            entity_counts[entity_name] = entity_counts.get(entity_name, 0) + count

        if not entity_counts:
            print("[get_entity_from_bbox] No entities found in bbox")
            return None

        # 5b. Compute occupancy ratios and filter
        candidates = []
        for entity_name, pixel_count in entity_counts.items():
            ratio = pixel_count / bbox_area
            base_name = extract_base_name(entity_name)
            name_match = (target_entity_name is None or base_name == target_entity_name)
            candidates.append({
                "entity_name": entity_name,
                "pixel_count": pixel_count,
                "ratio": ratio,
                "name_match": name_match,
            })
            print(f"  {entity_name}: {pixel_count} pixels, ratio={ratio:.3f}, name_match={name_match}")

        # 6b. Prefer name matches, then sort by occupancy ratio
        matched = [c for c in candidates if c["name_match"]]
        if matched:
            best = max(matched, key=lambda x: x["ratio"])
        else:
            best = max(candidates, key=lambda x: x["ratio"])

        # 7b. Threshold check
        if best["ratio"] < min_ratio:
            print(f"[get_entity_from_bbox] Best ratio {best['ratio']:.3f} below threshold {min_ratio}")
            return None

        print(f"[get_entity_from_bbox] Selected entity: {best['entity_name']} (ratio={best['ratio']:.3f})")
        return best["entity_name"]
