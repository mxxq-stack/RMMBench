"""
Metrics Module
Metric computation module:
1. RSTracker -- trajectory collector for the RS reflection metric (feed on_skill per skill, call result at episode end)
2. build_rs_tracker -- assembles RSTracker's two injected functions from env (grasp detection / read-only condition query)
3. Metric computation functions -- get_objective_score(ASbbox) / compute_rr_act(RRact) / compute_crnp(CRm/CRn/NP),
   called by codelab_eval at wrap-up (pure computation, does not assemble results); the remaining metrics SR/EE
   are computed in place on the eval side
"""
import math
import itertools
import copy
from typing import List, Dict, Any, Optional, Callable
import os
import json
from datetime import datetime
from typing import Dict
import numpy as np

from RMMBench.utils.utils import extract_base_name, get_hanan_l1_distance

# These skills do not perform an actual physical intervention and are not counted in the in-segment action count n (nor trigger grasp detection)
NO_COUNT = ("observe",)

# After these three skills execute, is_grasped is necessarily false; they perform no detection themselves and detection stops once they finish
STOP_DETECT = ("place",
               "open_door",
               "open_gripper"
               )


class RSTracker:
    """Trajectory collector for the RS reflection metric.

    Call on_skill once after each skill executes; internally it automatically maintains:
    - Object ownership: from a pick (inclusive) onward, keep watching that object by match; a mid-segment
      is_grasped=false is judged a drop; detection stops after place/open_door/open_gripper;
      the next pick starts detection again by the new match.
    - Segment splitting and settlement: a segment runs from pick a to the next pick of a different object or end;
      repeated picks of the same object are in-segment retries (pick_count accumulates, no segment split).
      The condition is queried only at settlement (i.e. final success at segment end), scored by four rules:
      0) condition is None (the object participates in no condition) -> None (excluded, so unrelated objects don't drag the score to zero)
      1) picked once and condition failed -> 0.0 (counted in M, should have reflected but didn't)
      2) picked once and condition succeeded -> None (excluded)
      3) picked multiple times -> (1.0 if condition else alpha) * exp(-lam * n)
         where n = number of actions after the first is_grasped=false in the segment
         (the false step itself is not counted; skills on the NO_COUNT list such as observe are not counted)

    Injected dependencies (no real physics engine required; unit tests can mock these two functions):
    - check_grasp(match) -> bool: after a pick, detect skill-by-skill whether the object is still grasped;
    - check_cond(match) -> Optional[bool]: read-only query of a condition's final success at segment settlement,
      None = the object participates in no condition; the query must be side-effect-free (must not advance
      internal condition state), guaranteed by the injector (see build_rs_tracker).
    match supports both entity-level ("lemon_0") and body-level ("stove_0/knob_front_right") formats.
    """

    def __init__(self, check_grasp, check_cond, lam=0.2, alpha=0.5):
        self.check_grasp = check_grasp   # fn(match) -> bool
        self.check_cond = check_cond     # fn(match) -> Optional[bool]
        self.lam, self.alpha = lam, alpha
        self.tracking = False            # whether an is_grasped check is currently in progress
        self.seg = None                  # currently open segment
        self.closed = []                 # already settled segments

    # ---------------- Public API ----------------

    def on_skill(self, action, match_entity=None):
        """Call once after each skill executes; match_entity is passed only for pick
        (the raw match string mapped from the bbox: entity-level "lemon_0", body-level "stove_0/knob_front_right")."""
        if action == "pick":
            if self.seg is not None and match_entity != self.seg["obj"]:
                self._close()            # pick b of a different object: settle the previous segment
            if self.seg is None:
                self.seg = {"obj": match_entity, "pick_count": 0, "steps": 0, "first_false": None}
            self.seg["steps"] += 1
            self.seg["pick_count"] += 1
            self.tracking = True
            self._detect()               # pick itself is also checked
        elif action == "end":
            self._close()                # end also counts as a settlement
            self.tracking = False
        elif action in NO_COUNT:
            return                       # skills on the list neither count steps nor get checked; n is unaffected by them
        else:
            if self.seg is not None:
                self.seg["steps"] += 1
            if not self.tracking:
                return
            if action in STOP_DETECT:
                self.tracking = False    # no detection; stop once execution finishes
            else:
                self._detect()

    def result(self):
        """Call after the episode ends; returns (RS mean or None, per-segment details).

        Segment detail fields: obj / pick_count / steps / first_false / cond / n / score;
        segments with a None score are excluded from the mean (the "excluded" of rules 0 and 2).
        """
        self._close()                    # Safety net: closes the last segment even if end was missed
        scores = [s["score"] for s in self.closed if s["score"] is not None]
        rs = sum(scores) / len(scores) if scores else None
        return rs, self.closed

    # ---------------- Internal methods ----------------

    def _detect(self):
        """Run one is_grasped check on the currently owned object; only records the position of the first false."""
        if self.seg["first_false"] is not None:
            return                       # already dropped, no need to check again
        if not self.check_grasp(self.seg["obj"]):
            self.seg["first_false"] = self.seg["steps"]

    def _close(self):
        """Settle the currently open segment: query the condition's final success and score it."""
        if self.seg is None:
            return
        seg, self.seg = self.seg, None
        cond = self._cond(seg["obj"])
        seg["cond"] = cond
        if cond is None:
            # The object participates in no condition, so there is no "should-be-satisfied" state; the whole segment is excluded
            seg["n"], seg["score"] = None, None
        elif seg["pick_count"] == 1 and not cond:
            # seg["n"], seg["score"] = None, 0.0       # Rule 1: failed with a single pick, score zero
            seg["n"], seg["score"] = None, None
        elif seg["pick_count"] == 1:
            seg["n"], seg["score"] = None, None      # Rule 2
        else:
            n = seg["steps"] - seg["first_false"] if seg["first_false"] is not None else 0
            seg["n"] = n
            seg["score"] = (1.0 if cond else self.alpha) * math.exp(-self.lam * n)
            # Rule 3 (gamma is 1.0 / alpha depending on the condition's success)
        self.closed.append(seg)

    def _cond(self, match_name):
        """Read-only query of the object's final condition success; None means it participates in no condition.

        Now wired to the injected check_cond (in build_rs_tracker, implemented by
        ConditionSet.get_entity_condition_status: the raw match string is passed through,
        "a/b" is matched via is_open prefix matching, while contain uses exact matching so there is no false hit).
        """
        return self.check_cond(match_name)


def build_rs_tracker(env, lam=0.2, alpha=0.5):
    """Build an RSTracker from env, wiring the two injected functions to the eval side.

    - check_grasp: entity-level uses entities[name].is_grasped(physics, robot);
      body-level (containing "/") and entities that cannot be found / lack is_grasped skip detection
      and always return True (treated as not dropped; first_false stays unset and n stays 0).
    - check_cond: passes the raw match string through to conditions.get_entity_condition_status's
      read-only query (does not advance AsynSequenceCondition / IsOpenCondition internal state);
      returns None when there are no conditions.
    """
    task = env.task

    def check_grasp(match_name):
        if "/" in match_name:            # body-level target, cannot detect, skip
            return True
        entities = getattr(task, "entities", None)
        entity = entities.get(match_name) if entities else None
        if entity is None or not hasattr(entity, "is_grasped"):
            return True                   # entity not found / no detection capability, skip
        return bool(entity.is_grasped(env.physics, task.robot))

    def check_cond(match_name):
        conditions = getattr(task, "conditions", None)
        if not conditions:
            return None
        return conditions.get_entity_condition_status(env.physics, match_name)

    return RSTracker(check_grasp=check_grasp, check_cond=check_cond,
                     lam=lam, alpha=alpha)


# ============================================================
# Metric computation functions (called by vlm_evaluator's evaluation methods)
# ============================================================

def get_objective_score(iou_list, threshold=0):
    """ASbbox: average over all bbox requests (with threshold=0, zero-score items are not dropped and count in the denominator)"""
    if not iou_list:
        return None
    ious = np.array(iou_list)
    valid_mask = ious > threshold
    score = np.sum(ious * valid_mask) / len(ious)
    return float(score)


# Navigation skills that are excluded from RRact skill recall in composite navigation tasks
NAVIGATION_SKILLS = {
    'moveforward', 'rotate_right', 'rotate_left', 'observe',
    'look_right', 'look_left', 'look_forward',
}


def compute_rr_act(vlm_seq: List[dict], expert_seq: List[dict],
                   task_type: Optional[str] = None, filter_nav: bool = False) -> float:
    """
    RRact skill recall rate (formerly skill_param_match_score), called by codelab_eval at wrap-up.

    Uses parameterized longest in-order matching (_calculate_param_match_len) to measure how much of the
    expert manipulation sequence can be recalled in order by the VLM sequence, converted into a 0~1 progress score.

    Args:
        vlm_seq / expert_seq: [{skill_name, params}, ...] raw skill sequences
        task_type: manipulation subtype; when 'flexible', enumerate the observe-free combinations of the expert
            sequence and take the max match score across combinations; otherwise treated as strict with direct matching
        filter_nav: set True for composite navigation tasks -- before matching, remove navigation skills from
            both sides, keeping only manipulation skills

    Returns:
        float: 0~1; 0.0 when either sequence is empty (or empty after filtering)
    """
    if filter_nav:
        expert_seq = [
            s for s in expert_seq
            if isinstance(s, dict) and s.get('skill_name') not in NAVIGATION_SKILLS
        ]
        vlm_seq = [
            s for s in vlm_seq
            if isinstance(s, dict) and s.get('skill_name') not in NAVIGATION_SKILLS
        ]

    if not expert_seq or not vlm_seq:
        return 0.0

    if task_type == 'flexible':
        # flexible: enumerate observe-free combinations of the expert sequence (observe/end split into blocks
        # + full permutations of pick-place blocks); combinations themselves contain no observe; take the max
        # match score across combinations.
        # Note: observe must not be filtered out of expert_seq in advance -- it is also a block boundary for flexible
        best_score = 0.0
        for expert_combination in generate_all_expert_combinations_no_observe(expert_seq):
            best_score = max(best_score, _calculate_param_match_len(vlm_seq, expert_combination))
        return best_score

    # strict: direct parameterized longest in-order matching; observe is an optional demonstrative step,
    # removed from both sides -- it takes no share of the denominator and does not participate in matching
    # (consistent with the composite filter_nav convention)
    expert_seq = [s for s in expert_seq if isinstance(s, dict) and s.get('skill_name') != 'observe']
    vlm_seq = [s for s in vlm_seq if isinstance(s, dict) and s.get('skill_name') != 'observe']
    return _calculate_param_match_len(vlm_seq, expert_seq)


def _check_subsequence_with_params(expert_seq: List[Dict], vlm_seq: List[Dict]) -> bool:
    """
    Enhanced subsequence check:
    1. Uses regex to clean object names (resolves lemon_0 vs lemon)
    2. Compatible with actions without a target (e.g. pull, push, end)
    """
    if not expert_seq: return True
    if not vlm_seq: return False

    # Turn vlm_seq into an iterator to ensure matching order is forward-only (irreversible)
    it = iter(vlm_seq)

    for expert_step in expert_seq:
        found_match = False

        # 1. Prepare expert-side data
        gold_name = expert_step.get('skill_name')
        # Parameter key alignment: open_/turn etc. use target_entity_name, place uses target_container_name,
        # part-level pick uses body_name (e.g. 'sink_0/handle')
        e_params = expert_step.get('params', {})
        gold_raw = str(
            e_params.get('target_entity_name')
            or e_params.get('target_container_name')
            or e_params.get('body_name')
            or ''
        )
        # Extract base name via regex: 'lemon_0' -> 'lemon'
        gold_target_base = extract_base_name(gold_raw)
        # Entity segment: 'sink_0/handle' -> 'sink_0' -> 'sink', allowing part-level and entity-level to match each other
        gold_entity_base = extract_base_name(gold_raw.split('/', 1)[0]) if gold_raw else ''

        # 2. Search the VLM sequence for a match
        for vlm_step in it:
            vlm_name = vlm_step.get('skill_name')
            # Prefer the entity name resolved from the bbox, matched_target (written back during pick/place execution);
            # fall back to the VLM's raw target output when absent
            vlm_raw = str(vlm_step.get('matched_target') or vlm_step.get('target') or '')
            vlm_target_base = extract_base_name(vlm_raw)
            vlm_entity_base = extract_base_name(vlm_raw.split('/', 1)[0]) if vlm_raw else ''

            # A. Skill names must be identical
            name_match = (vlm_name == gold_name)

            # B. Parameter matching logic:
            # passes by default when the expert step requires no object; when it does, either the full form matches
            # ('sink_0/handle' == 'sink_0/handle') or the owning entity matches
            # ('sink_0/handle' and 'sink_0' recognize each other)
            if not gold_target_base:
                param_match = True
            else:
                param_match = (gold_target_base == vlm_target_base) or (
                    gold_entity_base != '' and gold_entity_base == vlm_entity_base
                )

            # Both name and parameters match
            if name_match and param_match:
                found_match = True
                break

        # The current expert step found no match in the remaining VLM sequence: subsequence check failed
        if not found_match:
            return False

    return True


def _calculate_param_match_len(vlm_seq: List[Dict], expert_seq: List[Dict]) -> int:
    """
    Compute the LCS, compatible with the case where only some skills carry parameters
    """
    if not vlm_seq or not expert_seq:
        return 0

    m, n = len(expert_seq), len(vlm_seq)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            e_step = expert_seq[i - 1]
            v_step = vlm_seq[j - 1]

            # 1. Action name match (base criterion)
            name_match = (e_step.get('skill_name') == v_step.get('skill_name'))

            # 2. Parameter matching logic
            # Expert-side parameter reading (key alignment): open_/turn etc. use target_entity_name, place uses
            # target_container_name, part-level pick uses body_name (e.g. 'sink_0/handle'); then take the base via
            # regex ('lemon_0' -> 'lemon').
            e_params = e_step.get('params', {})
            e_raw = str(
                e_params.get('target_entity_name')
                or e_params.get('target_container_name')
                or e_params.get('body_name')
                or ''
            )
            e_base = extract_base_name(e_raw)
            # Entity segment: 'sink_0/handle' -> 'sink_0' -> 'sink', allowing part-level and entity-level to match each other
            e_entity = extract_base_name(e_raw.split('/', 1)[0]) if e_raw else ''

            # VLM-side parameter reading: prefer the entity name resolved from the bbox, matched_target
            # (written back during pick/place execution); when absent, fall back to the VLM's raw target output
            # (e.g. 'handle'), then take the base
            v_raw = str(v_step.get('matched_target') or v_step.get('target') or '')
            v_base = extract_base_name(v_raw)
            v_entity = extract_base_name(v_raw.split('/', 1)[0]) if v_raw else ''

            # --- Key decision logic ---
            if not e_base:
                # The expert defined no target object (e.g. pull, push, end, moveforward);
                # as long as the action name matches, parameter matching defaults to True
                param_match = True
            else:
                # Either the full form matches ('sink_0/handle' == 'sink_0/handle'),
                # or the owning entity matches ('sink_0/handle' and 'sink_0' recognize each other)
                param_match = (e_base == v_base) or (e_entity != '' and e_entity == v_entity)

            # 3. State transition
            if name_match and param_match:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])

        prev, curr = curr, prev
    score = prev[n] / len(expert_seq)
    # return prev[n]
    return float(score)


def generate_all_expert_combinations_no_observe(expert_seq):
    # 1. Physical blocking: bundle [pick, place] and skip observe and end directly
    blocks = []
    current_block = []

    for skill in expert_seq:
        if skill['skill_name'] in ['pick', 'place']:
            current_block.append(skill)

        # When hitting a node (observe or end), the current [pick, place] block ends
        if skill['skill_name'] in ['observe', 'end']:
            if current_block:
                blocks.append(current_block)
                current_block = []

    # 2. Permute all action blocks
    expert_combinations = []
    for perm in itertools.permutations(blocks):
        new_seq = []

        # 3. Concatenate all blocks in order
        for block in perm:
            new_seq.extend(copy.deepcopy(block))

        # 4. Append a single end at the very end of the sequence
        new_seq.append({'skill_name': 'end', 'params': {}})

        expert_combinations.append(new_seq)

    return expert_combinations


# ============================================================
# Composite navigation metrics (read purely from the env side, independent of skill sequences)
# ============================================================

def compute_crnp(env) -> Dict[str, Any]:
    """
    CRm / CRn / NP for composite navigation tasks (formerly manipulation_completion_rate /
    navigation_completion_rate / path_completion_rate); read purely from the env side, called by codelab_eval.

    Layering is based on state snapshots maintained by env.task.conditions throughout the episode:
    - Manipulation conditions (excluding contain_robot_pose): condition_has_been_met boolean (the final state
      is exact, 0/1)
    - Navigation conditions (containing contain_robot_pose): max(historical boolean, historical peak instant score),
      capturing the 0/0.5/1 tiers of "arrived but orientation not aligned"

    NP path completion rate:
    - Expert path = sum of per-segment Hanan L1 distances of init -> stop0 -> stop1 -> ...
    - A stop point counts as visited if its navigation condition's condition_progress_scores >= 0.5
    - Find the first unvisited stop point k; remaining distance = final_pos->stop_k + stop_k->stop_{k+1} + ...
    - path_completion = (expert path - remaining distance) / expert path

    Returns:
        dict: keys aligned with the seven metrics keys
            - cr_manipulation: manipulation condition completion rate (0.0~1.0)
            - cr_navigation: mean of navigation stop-point completion (mean of 0/0.5/1)
            - path_completion: NP path completion rate (0.0~1.0)
    """
    manipulation_scores = []
    navigation_scores = []

    # --- CRm / CRn: read env.task.conditions snapshots, layered by asyn_sequence ---
    condition_progress_scores = []
    asyn_sequence = []
    if env.task.conditions is not None:
        # Historical peak instant score: persisted every step during the episode, not lost when the robot leaves
        condition_progress_scores = env.task.conditions.condition_progress_scores
        condition_has_been_met = env.task.conditions.condition_has_been_met
        asyn_sequence = (env.task.config or {}).get("task", {}).get("conditions", {}).get("asyn_sequence", [])

        for i, has_been_met in enumerate(condition_has_been_met):
            # Determine whether this ConditionSet is a navigation type (containing contain_robot_pose)
            step_config = asyn_sequence[i] if i < len(asyn_sequence) else {}
            if "contain_robot_pose" in step_config:
                navigation_scores.append(max(1.0 if has_been_met else 0.0, condition_progress_scores[i]))
            else:
                manipulation_scores.append(1.0 if has_been_met else 0.0)

    cr_manipulation = (
        sum(manipulation_scores) / len(manipulation_scores) if manipulation_scores else 0.0
    )
    cr_navigation = (
        sum(navigation_scores) / len(navigation_scores) if navigation_scores else 0.0
    )

    # --- NP path completion rate: how much of the expert path was traversed ---
    path_completion = 0.0

    if hasattr(env.task, 'config_manager'):
        config_manager = env.task.config_manager
        if (hasattr(env.task, 'init_robot_info') and hasattr(env.task, 'robot_end_pos')
                and getattr(config_manager, 'target_object_info', None)):

            init_pos = env.task.init_robot_info["position"][:2]
            final_pos = env.task.robot_end_pos[:2]

            # Obstacle bboxes and boundary (for Hanan L1 path distance computation)
            bbox_list = []
            boundary = None
            if hasattr(env.task, 'get_robocasa_scene_class') and env.task.get_robocasa_scene_class is not None:
                try:
                    nav_bbox_info = env.task.get_robocasa_scene_class.get_obstacles_bbox(env.physics)
                    bbox_list = nav_bbox_info.get("bbox_list", [])
                    boundary = nav_bbox_info.get("boundary", None)
                except Exception:
                    pass

            # Extract the xy coordinates of each stop point in target_object_info order
            stop_positions = [info["position"][:2] for info in config_manager.target_object_info]
            n_stops = len(stop_positions)

            # Filter out the historical peak instant scores of navigation conditions, preserving order
            nav_progress_scores = []
            if env.task.conditions is not None:
                for i, step_cfg in enumerate(asyn_sequence):
                    if "contain_robot_pose" in step_cfg:
                        nav_progress_scores.append(condition_progress_scores[i])

            # Full expert path length: init -> stop0 -> stop1 -> ...
            waypoints = [init_pos] + stop_positions
            expert_path_len = sum(
                get_hanan_l1_distance(waypoints[j], waypoints[j + 1], bbox_list, boundary)
                for j in range(len(waypoints) - 1)
            )

            if expert_path_len <= 0:
                path_completion = 1.0
            else:
                # Find the first stop point whose position was not reached (historical peak score < 0.5)
                first_unvisited = n_stops  # defaults to all reached
                for k in range(n_stops):
                    pos_score = nav_progress_scores[k] if k < len(nav_progress_scores) else 0.0
                    if pos_score < 0.5:
                        first_unvisited = k
                        break

                if first_unvisited == n_stops:
                    # All stop points have been reached
                    path_completion = 1.0
                else:
                    # Remaining distance: final_pos -> stop_{first_unvisited} -> ... -> stop_{n-1}
                    remaining_waypoints = [final_pos] + stop_positions[first_unvisited:]
                    remaining_len = sum(
                        get_hanan_l1_distance(remaining_waypoints[j], remaining_waypoints[j + 1], bbox_list, boundary)
                        for j in range(len(remaining_waypoints) - 1)
                    )
                    path_completion = max(0.0, (expert_path_len - remaining_len) / expert_path_len)

    print(f"[CRm] {cr_manipulation:.3f} | [CRn] {cr_navigation:.3f} | [NP] {path_completion:.3f}")
    return {
        'cr_manipulation': cr_manipulation,
        'cr_navigation': cr_navigation,
        'path_completion': path_completion,
    }


# ============================================================
# Sequence matching utilities (backup, currently no callers)
# ============================================================

def evaluate_topology_match(vlm_seq: List[str], expert_seq: List[str]) -> Optional[float]:
    """
    Topology-based sequence matching evaluation

    Checks whether vlm_seq contains expert_seq as a subsequence

    Args:
        vlm_seq: Sequence of skill names executed by the VLM
        expert_seq: Sequence of expert skill names

    Returns:
        float: Topology match score (0.0~1.0), or None if it cannot be computed
    """
    if not expert_seq:
        return 1.0 if not vlm_seq else 0.0

    if not vlm_seq:
        return 0.0

    # Use LCS length as the topology match score
    lcs_length = _longest_common_subsequence_length(vlm_seq, expert_seq)
    return lcs_length / len(expert_seq)


def evaluate_skill_params(vlm_seq: List[Dict], expert_seq: List[Dict]) -> Dict[str, Any]:
    """
    Evaluate the degree of skill parameter matching

    Args:
        vlm_seq: Skill sequence executed by the VLM (with parameters)
        expert_seq: Expert skill sequence (with parameters)

    Returns:
        dict: Parameter matching related metrics
    """
    vlm_params = [s.get('params', {}) if isinstance(s, dict) else {} for s in vlm_seq]
    expert_params = [s.get('params', {}) if isinstance(s, dict) else {} for s in expert_seq]

    # Compute the LCS of parameter matching
    matched_count = 0
    total_params = 0

    for i, (v_params, e_params) in enumerate(zip(vlm_params, expert_params)):
        if i < len(expert_seq):
            # Compare key parameters
            total_params += 1
            if _params_match(v_params, e_params):
                matched_count += 1

    return {
        'skill_param_match_score': matched_count / total_params if total_params else 0.0,
        'matched_params_count': matched_count,
        'total_params_count': total_params
    }


def _check_subsequence(subseq: List[str], seq: List[str]) -> bool:
    """Check whether subseq is a subsequence of seq"""
    if not subseq:
        return True
    if not seq:
        return False

    it = iter(seq)
    return all(item in it for item in subseq)


def _longest_common_subsequence_length(seq1: List[str], seq2: List[str]) -> int:
    """Compute the longest common subsequence length"""
    if not seq1 or not seq2:
        return 0

    m, n = len(seq1), len(seq2)
    # Use a rolling array to optimize space complexity
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev

    return prev[n]


def _params_match(vlm_params: Dict, expert_params: Dict) -> bool:
    """Check whether two parameter sets match"""
    # List of key parameters
    key_params = ['target_entity_name', 'target_container_name', 'bbox']

    for key in key_params:
        if key in expert_params:
            if key not in vlm_params or vlm_params[key] != expert_params[key]:
                return False

    return True

def save_results(results: Dict, save_path: str, vlm_name: str) -> str:

    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d")
    filename = f"{vlm_name}_interactive_eval_{timestamp}.json"
    filepath = os.path.join(save_path, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Results saved to: {filepath}")
    return filepath