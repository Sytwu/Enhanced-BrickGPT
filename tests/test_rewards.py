import pytest

from brickgpt.data import BrickStructure
from brickgpt.masking import MaskConditioningConfig, stack_views
from brickgpt.training import (
    RewardConfig, syntax_reward, overlap_penalty, stability_reward_from_structure,
    silhouette_iou_from_structure, normalize_clip_score, compute_reward, stepwise_rewards,
)
from brickgpt.training.grpo_masked import compute_stepwise_advantages
from brickgpt.training.rewards import _valid_bricks

MCFG = MaskConditioningConfig()
STEP_CFG = RewardConfig(use_multi_turn=True, use_iou=False, use_semantic=False)


def _structure(text):
    return BrickStructure(_valid_bricks(text), world_dim=MCFG.world_dim)


# --- syntax gate ---------------------------------------------------------------------------------

@pytest.mark.parametrize('text,expected', [
    ('2x2 (0,0,0)\n', 1.0),
    ('2x6 (0,0,0)\n2x6 (2,0,0)\n', 1.0),
    ('garbage', -1.0),
    ('', -1.0),
    ('3x3 (0,0,0)\n', -1.0),  # 3x3 is not in the brick library
])
def test_syntax_reward(text, expected):
    assert syntax_reward(text) == expected


def test_gate_short_circuits_to_minus_one():
    b = compute_reward('garbage', cfg=RewardConfig(), mask_cfg=MCFG)
    assert b.total == -1.0 and b.syntax_ok is False
    assert b.overlap is None and b.stability is None and b.iou is None


# --- overlap (D3.2) ------------------------------------------------------------------------------

def test_overlap_penalty_on_double_stack():
    # Two identical 2x2 bricks at the same voxels -> 4 cells with occupancy 2 -> penalty -4.
    assert overlap_penalty(_structure('2x2 (0,0,0)\n2x2 (0,0,0)\n')) == -4.0
    # Collision-free structure -> 0.
    assert overlap_penalty(_structure('2x2 (0,0,0)\n2x2 (2,0,0)\n')) == 0.0


# --- stability (overlap-independent; D3.3) -------------------------------------------------------

def test_stability_connected_and_floating():
    assert stability_reward_from_structure(_structure('2x6 (0,0,0)\n2x6 (2,0,0)\n')) == 1.0
    assert stability_reward_from_structure(_structure('2x6 (0,0,0)\n2x6 (2,0,1)\n')) == -1.0  # floating


def test_overlap_does_not_double_fail_stability():
    # Two colliding ground bricks: penalized by overlap, but still ground-connected -> stability +1.
    s = _structure('2x2 (0,0,0)\n2x2 (0,0,0)\n')
    assert overlap_penalty(s) == -4.0
    assert stability_reward_from_structure(s) == 1.0  # NOT double-penalized


# --- 3-view IoU + per-view routing (D3.4) --------------------------------------------------------

def test_iou_perfect_over_all_views():
    text = '2x6 (0,0,0)\n'
    target = stack_views(text, MCFG)            # [3,20,20]
    iou = silhouette_iou_from_structure(_structure(text), target, [True, True, True], MCFG)
    assert iou == pytest.approx(1.0)


def test_iou_averages_only_provided_views():
    target_text = '2x6 (0,0,0)\n'
    target = stack_views(target_text, MCFG)
    # A prediction matching the top view (2x6 footprint) but two layers tall: top IoU=1, but
    # front/side differ. Routing to top-only must give 1.0; all-views must give < 1.0.
    pred = _structure('2x6 (0,0,0)\n2x6 (0,0,1)\n')
    top_only = silhouette_iou_from_structure(pred, target, [True, False, False], MCFG)
    all_views = silhouette_iou_from_structure(pred, target, [True, True, True], MCFG)
    assert top_only == pytest.approx(1.0)
    assert all_views < 1.0


def test_iou_none_when_no_view_provided():
    target = stack_views('2x6 (0,0,0)\n', MCFG)
    assert silhouette_iou_from_structure(_structure('2x6 (0,0,0)\n'), target, [False, False, False], MCFG) is None


# --- semantic (D3.5) -----------------------------------------------------------------------------

def test_normalize_clip_score():
    cfg = RewardConfig(clip_lo=0.15, clip_hi=0.35)
    assert normalize_clip_score(0.15, cfg) == 0.0
    assert normalize_clip_score(0.35, cfg) == 1.0
    assert normalize_clip_score(0.25, cfg) == pytest.approx(0.5)
    assert normalize_clip_score(0.05, cfg) == 0.0   # clipped
    assert normalize_clip_score(0.45, cfg) == 1.0   # clipped


def test_semantic_term_uses_precomputed_clip(monkeypatch):
    text = '2x6 (0,0,0)\n'
    target = stack_views(text, MCFG)
    cfg = RewardConfig(use_semantic=True)
    with_clip = compute_reward(text, target, [True, True, True], cfg=cfg, mask_cfg=MCFG, clip_score=0.25)
    assert with_clip.semantic == pytest.approx(0.5)
    # No clip score -> semantic term skipped even when enabled.
    no_clip = compute_reward(text, target, [True, True, True], cfg=cfg, mask_cfg=MCFG, clip_score=None)
    assert no_clip.semantic is None


# --- total + routing -----------------------------------------------------------------------------

def test_total_reward_perfect_match():
    text = '2x6 (0,0,0)\n'
    target = stack_views(text, MCFG)
    b = compute_reward(text, target, [True, True, True], cfg=RewardConfig(), mask_cfg=MCFG)
    # overlap 0 + stability +1 + iou 1.0 = 2.0
    assert b.total == pytest.approx(2.0)
    assert b.overlap == 0.0 and b.stability == 1.0 and b.iou == pytest.approx(1.0)


def test_total_reward_routes_out_iou_without_views():
    text = '2x6 (0,0,0)\n'
    target = stack_views(text, MCFG)
    with_views = compute_reward(text, target, [True, True, True], cfg=RewardConfig(), mask_cfg=MCFG)
    no_views = compute_reward(text, target, [False, False, False], cfg=RewardConfig(), mask_cfg=MCFG)
    assert no_views.iou is None
    assert with_views.total - no_views.total == pytest.approx(1.0)  # exactly w_iou * IoU(=1)


def test_reward_config_rejects_multiturn_with_terminal_terms():
    with pytest.raises(ValueError):
        RewardConfig(use_multi_turn=True, use_iou=True)
    with pytest.raises(ValueError):
        RewardConfig(use_multi_turn=True, use_iou=False, use_semantic=True)
    # multi-turn with only step-computable terms is allowed.
    RewardConfig(use_multi_turn=True, use_iou=False, use_semantic=False)


def test_breakdown_components_excludes_none():
    text = '2x6 (0,0,0)\n'
    target = stack_views(text, MCFG)
    b = compute_reward(text, target, [True, True, True], cfg=RewardConfig(), mask_cfg=MCFG)
    comps = b.components()
    assert set(comps) == {'overlap', 'stability', 'iou'}   # semantic off -> excluded


# --- multi-turn: per-brick step rewards (step-only) ----------------------------------------------

def test_stepwise_two_stable_bricks():
    # Stability is terminal: only the last committed brick carries it; overlap stays per-brick.
    steps = stepwise_rewards(['2x6 (0,0,0)', '2x6 (2,0,0)'], STEP_CFG, MCFG)
    assert [s.syntax_ok for s in steps] == [True, True]
    assert [s.overlap for s in steps] == [0.0, 0.0]
    assert [s.stability for s in steps] == [None, 1.0]
    assert [s.total for s in steps] == pytest.approx([0.0, 1.0])


def test_stepwise_incremental_overlap():
    # Second identical brick adds 4 overlapping voxels -> Δoverlap = -4; terminal stability +1 on it.
    steps = stepwise_rewards(['2x2 (0,0,0)', '2x2 (0,0,0)'], STEP_CFG, MCFG)
    assert steps[0].total == pytest.approx(0.0)     # overlap 0, stability deferred to terminal
    assert steps[1].overlap == pytest.approx(-4.0)
    assert steps[1].stability == 1.0
    assert steps[1].total == pytest.approx(-3.0)    # -4 (Δoverlap) + 1 (terminal stability)


def test_stepwise_invalid_line_gated_and_not_added():
    steps = stepwise_rewards(['2x6 (0,0,0)', 'garbage', '2x6 (2,0,0)'], STEP_CFG, MCFG)
    assert steps[1].total == -1.0 and steps[1].syntax_ok is False
    assert steps[1].overlap is None and steps[1].stability is None
    # The third brick is scored against {brick0, brick2} -- not poisoned by the bad middle line, and
    # carries the terminal stability (it is the last committed brick).
    assert steps[2].syntax_ok is True
    assert steps[2].overlap == pytest.approx(0.0)
    assert steps[2].stability == 1.0


def test_stepwise_floating_brick():
    steps = stepwise_rewards(['2x6 (0,0,0)', '2x6 (2,0,1)'], STEP_CFG, MCFG)
    assert steps[0].stability is None              # stability deferred to the terminal step
    assert steps[1].stability == -1.0              # final structure has a floating layer-1 brick
    assert steps[1].total == pytest.approx(-1.0)


def test_stepwise_terminal_stability_is_order_robust():
    # Roof-first build: the layer-1 brick is placed before its layer-0 support. A per-step check would
    # wrongly flag the first prefix as floating; the terminal check only judges the *final* structure,
    # which is fully grounded -> +1 on the last brick, 0 on the first.
    steps = stepwise_rewards(['2x6 (0,0,1)', '2x6 (0,0,0)'], STEP_CFG, MCFG)
    assert steps[0].stability is None and steps[0].total == pytest.approx(0.0)
    assert steps[1].stability == 1.0 and steps[1].total == pytest.approx(1.0)


def test_stepwise_skips_empty_lines():
    steps = stepwise_rewards(['2x6 (0,0,0)', '', '2x6 (2,0,0)'], STEP_CFG, MCFG)
    # An empty line cannot parse -> gated -1, but does not corrupt the running structure.
    assert [s.syntax_ok for s in steps] == [True, False, True]
    assert steps[2].stability == 1.0


# --- multi-turn: per-step (per-turn) advantages --------------------------------------------------

def test_stepwise_advantages_shapes_and_per_step_baseline():
    # Trajectories of lengths 3, 3, 1. gamma=1 -> G_t = sum of remaining step rewards.
    step_rewards = [[1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0]]
    A = compute_stepwise_advantages(step_rewards, gamma=1.0)
    assert [len(a) for a in A] == [3, 3, 1]                 # ragged shape preserved
    # Step index 0 standardized across all 3 trajectories -> column sums to ~0.
    assert A[0][0] + A[1][0] + A[2][0] == pytest.approx(0.0, abs=1e-6)
    # Step index 1: only trajectories 0 and 1 reach it -> 2-sample baseline, opposite signs.
    assert A[0][1] == pytest.approx(-A[1][1])


def test_stepwise_advantages_singleton_step_is_zero():
    # Only one trajectory reaches step index 1 -> no group signal -> advantage 0 there.
    A = compute_stepwise_advantages([[1.0, 2.0], [1.0]], gamma=1.0)
    assert A[0][1] == 0.0
