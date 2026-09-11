from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import pytest
import torch
from experiments.fmts_mainsteam_20260911.data import RichRecord, make_indices
from experiments.fmts_mainsteam_20260911.models import RichFusion, RichBlackbox
from experiments.fmts_mainsteam_20260911.spec import structured_specs
from experiments.fmts_mainsteam_20260911.run import day_mean, evaluate, main
from src.final_wm.synthetic import synthetic_canonical_arrays
from src.final_wm.contracts import FinalWMProtocolError

MAPPING = Path("configs/final_wm/channel_mapping_v2.json")


@pytest.fixture
def rich_record(tmp_path):
    arrays = synthetic_canonical_arrays(total_steps=2200, seed=11)
    gen = np.random.default_rng(10)
    centers = np.array([250,4,400,5,600,500,500,4,400])
    arrays["boundary_ext"] = (centers[None] * (1 + gen.normal(0,.01,(2200,9)))).astype("float32")
    arrays["aux"] = np.zeros((2200,15), dtype="float32")
    arrays["mill_on"] = np.zeros((2200,8), dtype="uint8")
    path = tmp_path / "synthetic_v22.npz"
    np.savez_compressed(path, **arrays)
    return RichRecord(path, MAPPING)


def test_future_extensions_do_not_enter_inputs_or_normalization(rich_record):
    r = rich_record
    idx = make_indices(r, True)["validation"][:1]
    before = r.batch(idx, 1, "cpu")
    mean, std = r.normalization()
    r.boundary_ext[idx.item():idx.item()+18] = 1e9
    after = r.batch(idx, 1, "cpu")
    assert torch.equal(before[1], after[1])
    mean2, std2 = r.normalization()
    assert torch.equal(mean, mean2) and torch.equal(std, std2)
    with pytest.raises(FinalWMProtocolError):
        r.candidates(2)


def test_invalid_history_and_time_gaps_excluded(rich_record):
    r = rich_record
    idx = int(r.candidates(1)[100])
    r.ext_valid[idx-1] = False
    assert idx not in r.candidates(1).tolist()
    r.ext_valid[idx-1] = True
    r.timestamps[idx-2] += 1
    assert idx not in r.candidates(1).tolist()


@pytest.mark.parametrize("arm", ["fusion_gru_norew", "fusion_token_xattn_norew"])
def test_extension_gradient_reaches_both_observers(rich_record, arm):
    torch.set_num_threads(1)
    r = rich_record
    mean, std = r.normalization()
    spec = next(s for s in structured_specs((0,)) if s.arm == arm)
    model = RichFusion(spec, mean, std)
    h, ext, a, b, y, days = r.batch(make_indices(r,True)["train"][:2],0,"cpu")
    anchor = model.base._steady_initial_state(h)
    observer = model.base.observer
    with torch.no_grad():
        observer.mu_head.weight.normal_(0,.01)
    ext.requires_grad_()
    mu, _ = observer.posterior(h.obs,h.actions,
        torch.cat([h.boundary,(ext-mean[-9:])/std[-9:]],-1),anchor)
    mu[:,3:7].sum().backward()
    assert ext.grad is not None and torch.isfinite(ext.grad).all()
    assert torch.all(ext.grad.abs().sum((0,1)) > 0)
    assert ((mu-anchor).abs() <= observer.state_scale*.1 + 1e-4).all()


def test_greybox_ignores_extension_and_rich_models_share_inputs(rich_record):
    torch.set_num_threads(1)
    mean,std = rich_record.normalization()
    spec = next(s for s in structured_specs((0,)) if s.arm == "greybox_steady_none")
    model = RichFusion(spec,mean,std).eval()
    h,ext,a,b,*_ = rich_record.batch(make_indices(rich_record,True)["validation"][:1],1,"cpu")
    with torch.no_grad():
        assert torch.equal(model(h,ext,a,b).temps_mu,model(h,ext+100,a,b).temps_mu)
        bb = RichBlackbox(mean,std).eval()
        assert not torch.equal(bb(h,ext,a,b),bb(h,ext+1,a,b))


def test_complete_smoke_cli_preserves_all_four_arms(rich_record,tmp_path,monkeypatch):
    torch.set_num_threads(1)
    out = tmp_path / "smoke"
    monkeypatch.setattr(sys,"argv",["run","--record",str(rich_record.path),
        "--mapping",str(MAPPING),"--out",str(out),"--smoke"])
    main()
    summary = json.loads((out / "summary.json").read_text())
    assert len(summary["runs"]) == 4
    assert all(r["status"] == "complete" for r in summary["runs"])
    assert summary["selected_hybrid"] == "INCOMPLETE"
    assert summary["locked_test_evaluated"] is False
    from experiments.fmts_mainsteam_20260911.audit import audit
    result = audit(out, rich_record.path)
    assert result["complete"] and result["array_sets_replayed"] == 8
    for report in summary["runs"]:
        run_dir = out / f"{report['arm']}_seed0"
        with np.load(run_dir / "prediction.npz") as raw:
            assert raw["prediction"].shape == (4,18)
        with np.load(run_dir / "response.npz") as raw:
            assert raw["valve1_support"].shape == (4,18)
    with pytest.raises(SystemExit):
        main()  # output overwrite is forbidden


def test_equal_day_weighting():
    values = np.array([1.,1.,7.])
    assert day_mean(values,np.array([1,1,2])) == 4.
