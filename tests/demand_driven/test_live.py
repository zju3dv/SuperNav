"""AI2-THOR spectator preserves policy pixels/actions and the shared web contract."""

from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from supernav.backends.ai2thor.native import NativeSession
from supernav.backends.ai2thor.protocol import NativeConfig
from supernav.web.store import SessionStore


class Controller:
    def __init__(self):
        self.calls = []
        self.last_event = SimpleNamespace(
            frame=np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8),
            metadata=dict(agent=dict(position=dict(x=0, y=.95, z=0),
                                     rotation=dict(x=0, y=0, z=0), cameraHorizon=0),
                          fov=120, lastActionSuccess=True, collided=False, objects=[]))

    def step(self, **kwargs):
        self.calls.append(kwargs)
        agent = self.last_event.metadata['agent']
        if kwargs['action'] == 'MoveAhead':
            agent['position']['z'] += .1
        elif kwargs['action'] == 'RotateRight':
            agent['rotation']['y'] += kwargs['degrees']
        return self.last_event


def make_session(tmp_path, name='episode'):
    episode = dict(episode_id='e', scene_id='train.jsonl_1', instruction='Find a drink',
                   start_position=dict(x=0, y=.95, z=0), start_rotation_y=0, start_horizon=0,
                   reproducibility={'house_data_sha256': 'hash'}, stage_plan=[])
    session = NativeSession(Controller(), episode, NativeConfig(), tmp_path / name)
    session.initialize()
    return session


def test_live_frames_pose_activity_and_unscored_claim_are_visible(monkeypatch, tmp_path):
    live = tmp_path / 'live'
    monkeypatch.setenv('SUPERNAV_LIVE_DIR', str(live))
    session = make_session(tmp_path)
    store = SessionStore(live)
    key = store.sessions()[0]['id']
    initial = store.snapshot(key)
    assert initial['scene'] == 'AI2-THOR · train.jsonl_1'
    assert initial['instruction'] == 'Find a drink'
    assert initial['pose']['heading_deg'] == 90  # Unity yaw 0 faces +Z.
    assert set(initial['panoramas'][-1]['views']) == {'front'}
    original_rgb = session.last_rgb.copy()
    session.step('forward')
    session.step('right')
    session.observe()  # Flush final tool-response pixels even between FPS ticks.
    moved = store.snapshot(key)
    assert moved['step'] == 2 and moved['pose']['position'][2] == pytest.approx(.1)
    assert moved['pose']['heading_deg'] == 80
    assert moved['path_length_m'] == pytest.approx(.1)
    assert moved['events'][-1]['action'] == 'RotateRight'
    assert moved['events'][-1]['ok'] is True
    result = session.claim(session.observation_id, 'Test claim, not a success', [.5, .5])
    assert result == {'recorded': True, 'success_scoring': 'withheld'}
    doc = store.snapshot(key)
    overlay = doc['overlays'][-1]
    assert overlay['image_ref'] == session.observation_id
    assert overlay['capture_seq'] == doc['panoramas'][-1]['capture_seq']
    assert doc['agent_claim'] is None and doc['evaluation'] is None
    front = store.frame(key, doc['panoramas'][-1]['id'], kind='panorama', view='front')
    assert np.array_equal(np.asarray(Image.open(front)), original_rgb)
    assert store.frame(key, doc['panoramas'][-1]['id'], kind='panorama', view='right') is None
    marked = np.asarray(Image.open(store.frame(key, overlay['id'], kind='overlay')))
    assert not np.array_equal(marked, original_rgb)
    assert np.array_equal(session.last_rgb, original_rgb)
    session.finish()
    closed = store.snapshot(key)
    assert closed['status'] == 'closed' and closed['step'] == 3
    assert closed['stop_called'] and closed['agent_claim'] is None
    assert len(session.controller.calls) == 3  # Teleport, forward, right; no spectator actions.


@pytest.mark.parametrize('broken', [False, True])
def test_disabled_or_broken_live_view_preserves_execution(monkeypatch, tmp_path, broken):
    target = tmp_path / 'occupied'
    target.write_text('not a directory')
    monkeypatch.setenv('SUPERNAV_LIVE_DIR', str(target) if broken else '')
    session = make_session(tmp_path)
    assert session.live.publisher is None
    result = session.step('forward')
    assert result['action_success'] and session.actions == 1
    assert len(session.controller.calls) == 2
    assert session.finish()['stop_called']


def test_disk_failure_after_start_does_not_retry_the_action(monkeypatch, tmp_path):
    monkeypatch.setenv('SUPERNAV_LIVE_DIR', str(tmp_path / 'live'))
    session = make_session(tmp_path)
    def fail(*args, **kwargs):
        raise OSError('disk unavailable')
    monkeypatch.setattr(session.live.publisher, 'publish', fail)
    assert session.step('forward')['action_success']
    assert len(session.controller.calls) == 2
    assert session.live.publisher is None
    assert not session.infra_failed


def test_final_budget_frame_and_interruption_are_published(monkeypatch, tmp_path):
    live = tmp_path / 'live'
    monkeypatch.setenv('SUPERNAV_LIVE_DIR', str(live))
    session = make_session(tmp_path)
    session.live.publisher.next_capture = float('inf')
    session.actions = session.config.max_actions - 1
    assert session.step('forward')['terminal']
    doc = SessionStore(live).snapshot(session.live.publisher.session_id)
    assert doc['status'] == 'closed' and not doc['stop_called']
    assert doc['step'] == 500 and doc['pose']['position'][2] == pytest.approx(.1)
    another = make_session(tmp_path, 'interrupted')
    another.finish('operator_interrupt')
    assert SessionStore(live).snapshot(another.live.publisher.session_id)['status'] == 'interrupted'
    assert another.live.publisher.session_id != session.live.publisher.session_id
