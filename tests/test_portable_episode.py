import json
import pytest
from joint_episode import load_episode

def test_relative_media_uses_episode_root(tmp_path, monkeypatch):
    episode=tmp_path/'episode';episode.mkdir()
    (episode/'bundle.json').write_text(json.dumps({'events':[], 'media':{'f000001':{'path':'frames/one.jpg'}}}))
    monkeypatch.chdir(tmp_path)
    assert load_episode(episode)['media']['f000001']['path']==str(episode/'frames/one.jpg')

def test_relative_media_cannot_escape_episode(tmp_path):
    (tmp_path/'bundle.json').write_text(json.dumps({'events':[], 'media':{'f000001':{'path':'../outside.jpg'}}}))
    with pytest.raises(ValueError,match='escapes'):
        load_episode(tmp_path)

def test_training_only_action_episode_needs_no_eval_cache(tmp_path):
    (tmp_path/'events.json').write_text('[]')
    (tmp_path/'media.json').write_text('{}')
    data=load_episode(tmp_path)
    assert data['events']==[] and data['actions']==[] and data['schedule']==[]
