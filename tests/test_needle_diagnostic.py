from tools.weight_space_lab.needle_diagnostic import ROOT, snapshot_names
from tools.weight_space_lab.sandbox import ROOT as SANDBOX_ROOT


def test_new_worktree_uses_its_own_jail_and_exact_tensor_allowlist():
    assert ROOT == SANDBOX_ROOT
    assert snapshot_names({'layers': [7, 19], 'modules': ['mlp.down_proj.weight']}) == [
        'model.language_model.layers.7.mlp.down_proj.weight',
        'model.language_model.layers.19.mlp.down_proj.weight']
