"""Frozen bundled observer repair; parent physics/data/training are unchanged."""
from dataclasses import asdict, replace
from experiments.fmts_mainsteam_20260911.spec import structured_specs

PROTOCOL_ID = 'FMTS-M7R1'
ARM = 'fusion_m7var_norew'
REFERENCE_ARMS = ('fusion_gru_norew', 'fusion_token_xattn_norew',
                  'blackbox_itransformer', 'greybox_steady_none')
MODEL = dict(
    implementation='M7Observer_v1', parent_template='fusion_gru_norew',
    history_channels=23, history_steps=96, patch_length=16, patch_stride=8,
    d_model=64, tcn_layers=2, attention_heads=4, dropout=0.1,
    context_widths=[1518, 256, 128], context_output_norm='non-affine LayerNorm',
    history_normalization='parent physical scales + train-only extension stats; then RevIN',
    absolute_context='concatenate pre-RevIN per-window mean and std, 23+23 channels',
    readout='independent 11-row linear heads + parent pressure features; zero mean initialization',
    correction='parent 0.1*state_scale*tanh; same hybrid slow-state mask',
    weights='fresh; do not import historical learned checkpoints',
)


def specs(seeds=(0, 1, 2)):
    # TrainSpec describes the unchanged physical/training template. The effective
    # observer is explicitly MODEL, not the template's gru dispatch identifier.
    return [replace(s, arm=ARM) for s in structured_specs(seeds)
            if s.arm == 'fusion_gru_norew']


def serialized_spec():
    return dict(protocol_id=PROTOCOL_ID, parent_protocol='fmts_mainsteam_20260911/v0.2',
                comparison_status='EXPLORATORY_POST_V02_BUNDLED_OBSERVER_REPAIR',
                scientific_change='M7-family encoder + independent vector readout; not a factorial ablation',
                model=MODEL, training_templates=[asdict(s) for s in specs()], seeds=[0, 1, 2],
                parent_windows='byte-identical indices.npz; no resampling',
                references=list(REFERENCE_ARMS), locked_test_enabled=False,
                response_used_for_selection=False, parent_results_retained=True,
                plant_response_truth=False, old_arms_retrained=False,
                mae_advance_rule='at least 2/3 paired wins and mean <= retained GRU; validation only',
                health='diagnostic only; no post-hoc checkpoint selection; manual review before adoption',
                greybox_followup='FMTS-GNR1 unchanged and independently registered')
