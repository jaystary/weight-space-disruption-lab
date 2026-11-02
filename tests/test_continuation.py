import pytest
from tools.weight_space_lab import continue_campaign

def test_existing_run_is_never_automatically_relaunched(tmp_path,monkeypatch):
    (tmp_path/'edit-1').mkdir();(tmp_path/'edit-1/chosen.json').write_text('{}')
    (tmp_path/'edit-2').mkdir()
    monkeypatch.setattr(continue_campaign,'RUNS',tmp_path)
    def forbidden(*args,**kwargs):raise AssertionError('duplicate experiment was dispatched')
    monkeypatch.setattr(continue_campaign.subprocess,'run',forbidden)
    with pytest.raises(RuntimeError,match='refusing duplicate'):continue_campaign.main()
