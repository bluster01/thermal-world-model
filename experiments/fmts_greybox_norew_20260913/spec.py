"""Single-scientific-change supplement; no mutation of the parent protocol."""
from dataclasses import asdict, replace
from experiments.fmts_mainsteam_20260911.spec import structured_specs

ARM='greybox_steady_none_norew'
PARENT_ARM='greybox_steady_none'
PROTOCOL_ID='FMTS-GNR1'


def specs(seeds=(0,1,2)):
    return [replace(s,arm=ARM,closure_mode='none_norew')
            for s in structured_specs(seeds) if s.arm==PARENT_ARM]


def serialized_spec():
    return {'protocol_id':PROTOCOL_ID,'parent_protocol':'fmts_mainsteam_20260911/v0.2',
            'scientific_change':'rewet_ablate false -> true; aW1/aW2 pinned near zero',
            'seeds':[0,1,2],'runs':[asdict(s) for s in specs()],
            'parent_windows':'byte-identical indices.npz; do not resample',
            'locked_test_enabled':False,'response_used_for_selection':False,
            'plant_response_truth':False,'comparison_status':'EXPLORATORY_POST_V02',
            'retain_parent_results':True,'rerun_parent_arms':False}
