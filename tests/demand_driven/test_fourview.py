"""Four-view geometry, reference preservation and physical-budget invariants."""
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
from supernav.backends.ai2thor.fourview import FourViewSession, configuration, VIEWS
from supernav.backends.ai2thor.protocol import horizontal_fov

class Controller:
    def __init__(self, bad=None):
        self.calls=[]
        self.bad=bad
        self.last_event=SimpleNamespace(metadata=dict(
            agent=dict(position=dict(x=0.,y=.95,z=0.),rotation=dict(x=0.,y=0.,z=0.),cameraHorizon=0.),
            fov=configuration().vertical_fov_deg, objects=[dict(objectId="Mug|1")],
            thirdPartyCameras=[],lastActionSuccess=True,collided=False))
        self.last_event.third_party_camera_frames=[]
    def step(self,**kw):
        self.calls.append(kw)
        event=copy.deepcopy(self.last_event)
        a=event.metadata["agent"];name=kw["action"]
        if name in ("AddThirdPartyCamera","UpdateThirdPartyCamera"):
            camera={k:copy.deepcopy(kw[k]) for k in ("position","rotation","fieldOfView")}
            if self.bad and len(event.metadata["thirdPartyCameras"])>=1:
                if self.bad=="fov":camera["fieldOfView"]+=1
                if self.bad=="pose":a["rotation"]["y"]+=1
            if name=="AddThirdPartyCamera":
                event.metadata["thirdPartyCameras"].append(camera)
                n=len(event.third_party_camera_frames)
                rgb=np.zeros((480,640,3),dtype=np.uint8);rgb[:,:320,n%3]=255
                event.third_party_camera_frames.append(rgb)
            else:event.metadata["thirdPartyCameras"][kw["thirdPartyCameraId"]]=camera
        elif name in ("RotateLeft","RotateRight"):
            a["rotation"]["y"]=(a["rotation"]["y"]+kw["degrees"]*(-1 if name=="RotateLeft" else 1))%360
        self.last_event=event
        return event

def make(tmp_path,bad=None,clock=lambda:100.):
    ep=dict(episode_id="e",scene_id="train.jsonl_1",instruction="Prepare a drink",
            start_position=dict(x=0.,y=.95,z=0.),start_rotation_y=0,start_horizon=0,
            reproducibility=dict(house_data_sha256="123"),
            stage_plan=[dict(allowed_target_candidates=[dict(object_id="Mug|1")])])
    return FourViewSession(Controller(bad),ep,configuration(),tmp_path/"episode",
                           house={"rooms":[{"floorPolygon":[{"y":0}]}]},clock=clock)

def test_four_views_are_pose_preserving_and_free(tmp_path):
    s=make(tmp_path);obs=s.initialize()
    assert list(obs["images"])==list(VIEWS) and s.actions==0
    assert horizontal_fov(s.config.vertical_fov_deg,640,480)==pytest.approx(90)
    assert s.last_rgb is not s.view_frames["front"]
    assert "Mug|1" not in json.dumps(obs)
    assert all("pose" not in k for k in obs)
    views=s.controller.last_event.metadata["thirdPartyCameras"]
    assert [v["rotation"]["y"] for v in views]==[0,90,180,270]
    assert all(v["position"]["y"]==1.25 for v in views)
    n=len(s.controller.calls);s.observe();assert len(s.controller.calls)==n

@pytest.mark.parametrize("bad",["fov","pose"])
def test_incorrect_side_camera_is_rejected(tmp_path,bad):
    with pytest.raises(RuntimeError):make(tmp_path,bad).initialize()

@pytest.mark.parametrize("view,count,yaw",[("right",9,90),("left",9,270),("back",18,180),("front",0,0)])
def test_side_goal_turns_are_charged_and_reference_is_preserved(tmp_path,view,count,yaw):
    s=make(tmp_path);obs=s.initialize();s.begin();goal=s.view_frames[view].copy();received=[]
    class Policy:
        def act(self,history,reference,pixel):
            received.append((history,reference,pixel))
            return dict(trajectories=[[[0,0]]*8],scores=[1],done_probability=.9,confidence=1)
    result=s.local_navigate(obs["observation_id"],[.3,.7],Policy(),3,view=view)
    assert s.actions==count and s.controller.last_event.metadata["agent"]["rotation"]["y"]==yaw
    assert len(received)==2 and np.array_equal(received[0][1],goal)
    assert result["reason"]=="local_policy_stop_not_task_success"
    assert not s.terminal

def test_stale_reference_cannot_move_or_claim(tmp_path):
    s=make(tmp_path);obs=s.initialize();s.begin();s.step("right");n=s.actions
    with pytest.raises(ValueError):s.local_navigate(obs["observation_id"],[.5,.5],None,view="left")
    with pytest.raises(ValueError):s.claim_view(obs["observation_id"],"mug",[.5,.5],"front")
    assert s.actions==n

def test_alignment_stops_at_budget_without_synthetic_stop(tmp_path):
    s=make(tmp_path);obs=s.initialize();s.begin();s.actions=497
    s.local_navigate(obs["observation_id"],[.5,.5],None,view="back")
    assert s.actions==500 and s.terminal
    status=json.loads((s.output/"status.json").read_text())
    assert status["reason"]=="execution_budget_exhausted" and not status["stop_called"]

def test_wall_deadline_blocks_physical_actions_and_stop(tmp_path):
    now=[100.];s=make(tmp_path,clock=lambda:now[0]);s.initialize();s.begin();now[0]=3701.
    before=len(s.controller.calls);s.step("right")
    assert len(s.controller.calls)==before and s.actions==0
    status=s.finish();assert status["reason"]=="wall_timeout" and not status["stop_called"]

def test_stop_cost_and_claim_view_do_not_use_gt_feedback(tmp_path):
    s=make(tmp_path);obs=s.initialize();s.begin()
    assert s.claim_view(obs["observation_id"],"a cup",[.5,.5],"right")==dict(recorded=True,success_scoring="withheld")
    assert s.actions==0
    stopped=s.finish();assert stopped["stop_called"] and stopped["execution_units"]==1



def test_fourview_protocol_is_distinct_and_not_reported_as_front_geometry():
    receipt = configuration().receipt()
    assert receipt['profile'] == 'neednav_fourview_unscored'
    assert receipt['observation_views'] == list(VIEWS)
    assert receipt['horizontal_fov_deg'] == pytest.approx(90)
    assert 'vertical_fov_deg' not in receipt['frozen_from_package']
    assert receipt['wall_budget_s'] == 3600 and receipt['success_scoring'] == 'withheld'


def test_web_receives_complete_current_groups_and_side_claim_pixels(tmp_path, monkeypatch):
    from PIL import Image
    from supernav.web.store import SessionStore

    live = tmp_path/'live'
    monkeypatch.setenv('SUPERNAV_LIVE_DIR', str(live))
    s=make(tmp_path);obs=s.initialize();s.begin()
    store=SessionStore(live);key=s.live.publisher.session_id
    group=store.snapshot(key)['panoramas'][-1]
    assert list(group['views']) == list(VIEWS)
    for view in VIEWS:
        assert group['views'][view]['image_ref'] == obs['view_refs'][view]
        assert np.array_equal(np.asarray(Image.open(store.frame(key,group['id'],kind='panorama',view=view))),s.view_frames[view])
    s.claim_view(obs['observation_id'],'Right-view test claim',[.3,.7],'right')
    state=store.snapshot(key);overlay=state['overlays'][-1]
    assert overlay['direction']=='right' and overlay['image_ref']==obs['view_refs']['right']
    assert overlay['capture_seq']==group['capture_seq']
    marked=np.asarray(Image.open(store.frame(key,overlay['id'],kind='overlay')))
    assert np.array_equal(marked[:20,:20],s.view_frames['right'][:20,:20])
    assert not np.array_equal(marked,s.view_frames['right'])
    assert state['agent_claim'] is None and state['evaluation'] is None
    assert s.actions==0 and len(s.controller.calls)==5
    for _ in range(14):
        s.step('right');s.observe()
    state=store.snapshot(key)
    assert len(state['panoramas'])==12
    assert len({g['capture_seq'] for g in state['panoramas']})==12
    assert all(list(g['views'])==list(VIEWS) for g in state['panoramas'])
    assert state['panoramas'][-1]['capture_seq']==s.capture
    assert state['pose']['heading_deg']==pytest.approx((90-140)%360)
    assert len(list(live.glob('*.pano-*.png')))==48
    assert s.actions==14 and len(s.controller.calls)==75


def test_side_goal_overlay_keeps_original_view_after_physical_alignment(tmp_path, monkeypatch):
    from supernav.web.store import SessionStore
    monkeypatch.setenv('SUPERNAV_LIVE_DIR',str(tmp_path/'live'))
    s=make(tmp_path);obs=s.initialize();s.begin()
    class Policy:
        def act(self,history,reference,pixel):
            return dict(trajectories=[[[0,0]]*8],scores=[1],done_probability=.9,confidence=1)
    s.local_navigate(obs['observation_id'],[.3,.7],Policy(),3,view='back')
    doc=SessionStore(tmp_path/'live').snapshot(s.live.publisher.session_id)
    assert s.actions==18 and len(doc['overlays'])==1
    assert doc['overlays'][0]['image_ref']==obs['view_refs']['back']
    assert doc['overlays'][0]['capture_seq']==1 and doc['overlays'][0]['direction']=='back'
    assert doc['panoramas'][-1]['capture_seq']==19


def test_real_http_and_mcp_transports_deliver_four_images(tmp_path, monkeypatch):
    import multiprocessing, os, socket, subprocess, sys, time, urllib.request
    from supernav.backends.ai2thor.cli import serve
    from supernav.paths import python_paths
    s=make(tmp_path);s.initialize();s.begin()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    process=multiprocessing.get_context('fork').Process(target=serve,args=(s,port,None))
    process.start()
    monkeypatch.setenv('PYTHONPATH',os.pathsep.join(python_paths()))
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        deadline=time.monotonic()+10
        while True:
            try:
                with opener.open(f'http://127.0.0.1:{port}/healthz',timeout=1) as response:
                    assert json.load(response)['views']==list(VIEWS)
                break
            except OSError:
                if time.monotonic()>=deadline:raise
                time.sleep(.05)
        result=subprocess.run([sys.executable,'-m','supernav.methods.demand_driven.check_mcp',
                               '--storage-root',str(tmp_path),'--python',sys.executable,
                               '--port',str(port),'--output',str(tmp_path/'mcp.json'),'--views','four'],
                              capture_output=True,text=True,timeout=30)
        assert result.returncode==0,result.stderr
        report=json.loads((tmp_path/'mcp.json').read_text())
        assert report['valid'] and report['observation_views']==list(VIEWS)
        status=json.loads((s.output/'status.json').read_text())
        assert status['stop_called'] and status['action_count']==2
        claim=json.loads((s.private/'claim_views.jsonl').read_text())
        assert claim['view']=='right'
    finally:
        process.terminate();process.join(timeout=5)
        if process.is_alive():process.kill();process.join(timeout=5)
