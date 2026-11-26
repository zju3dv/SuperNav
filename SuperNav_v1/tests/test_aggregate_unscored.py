"""Completion claims and missing scores retain their own meaning."""
import json
import sys

from supernav.evaluation.aggregate import main, summarize


def objective(success, threshold=.25):
    return {'success': success, 'spl': 1.0 if success else 0.0,
            'criterion': 'viewpoint_geodesic', 'success_distance_m': threshold,
            'l_source': 'viewpoint_geodesic', 'stop_required': False,
            'unreachable_freebie': True}


def run_cli(tmp_path, monkeypatch, rows):
    (tmp_path/'results.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    monkeypatch.setattr(sys, 'argv', ['aggregate', '--runs-dir', str(tmp_path)])
    assert main() == 0
    return json.loads((tmp_path/'summary.json').read_text())['arms']['test']


def test_unscored_result_stays_null(tmp_path, monkeypatch, capsys):
    result = run_cli(tmp_path, monkeypatch, [{'arm': 'test', 'success': None}])
    assert result['agent_completion_rate'] is None
    assert result['objective_sr'] is None and result['objective_spl'] is None
    assert result['objective_scored_n'] == 0 and result['objective_unscored_n'] == 1
    assert result['n'] == 1
    assert 'null' in capsys.readouterr().out


def test_completion_claim_is_not_objective_sr(tmp_path, monkeypatch):
    result = run_cli(tmp_path, monkeypatch, [{'arm': 'test', 'success': True,
        'success_semantics': 'returncode_zero_and_session_closed_and_structured_achieved',
        'llm_turns': 2}])
    assert result['agent_completion_rate'] == 1
    assert result['objective_sr'] is None and result['objective_spl'] is None
    assert result['llm_turns_mean'] == 2
    assert 'success_rate' not in result


def test_objective_failure_is_available_zero_even_with_achieved_claim():
    result = summarize([{'success': True, 'objective_score': objective(False)}])
    assert result['agent_completion_rate'] == 1
    assert result['objective_sr'] == 0 and result['objective_spl'] == 0
    assert result['objective_scored_n'] == 1 and result['objective_unscored_n'] == 0


def test_threshold_change_separates_scoring_profiles():
    rows = [{'objective_score': objective(True)}, {'objective_score': objective(False)}]
    assert summarize(rows)['objective_sr'] == .5
    rows[1]['objective_score']['success_distance_m'] = 1.0
    result = summarize(rows)
    assert result['objective_sr'] is None and result['objective_spl'] is None
    assert {p['profile']['success_distance_m']: p['sr'] for p in result['objective_profiles']} == {.25: 1, 1.0: 0}
    assert result['objective_scored_n'] == 2


def test_missing_evidence_preserves_total_and_reports_available_denominator():
    result = summarize([{'objective_score': objective(False)}, {'success': None}])
    assert result['n'] == 2 and result['objective_sr'] == 0
    assert result['objective_scored_n'] == 1 and result['objective_unscored_n'] == 1
    assert result['objective_profiles'][0]['sr_scored_n'] == 1
    assert summarize([{'objective_score': {'success': True}}])['objective_sr'] is None
