"""Synthetic manifests and bytes only: no study checkpoint or cloud access."""
from pathlib import Path
import copy
import json
import sys
import tarfile
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import package_local_goal_return_outputs as package


def routing():
    return dict(protocol_sha256=package.PROTOCOL_SHA, workers=[dict(
        condition=condition, name=f'ring-20261105-local-return-{label}-01', zone='test-zone',
        local_artifact_root=f'results/ring_cloud_artifacts/ring-20261105-local-return-{label}-01',
        artifact_subdir='experiment') for condition, label in [('gc100','grid'), ('pgc33','pgc')]])


def cleanup_fixture(root, route):
    rows=[]
    for worker in route['workers']+route.get('prior_attempts', []):
        row={key:worker[key] for key in ('condition','name','zone')}
        row.update(instance_absent=True, disk_absent=True, evidence=[])
        for kind in ('instances','disks'):
            relative=f"results/local_goal_return_design/cloud_audit/{row['name']}_{kind}_absence.log"
            path=root/relative;path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"Resource projects/test/zones/test-zone/{kind}/{row['name']} was not found\n")
            row['evidence'].append(dict(path=relative, sha256=package.sha(path)))
        rows.append(row)
    return dict(protocol_sha256=package.PROTOCOL_SHA, complete=True, verified_utc='synthetic', workers=rows)


@pytest.mark.parametrize('value', ['', '.', '/tmp/outside', '../outside', 'a/../../outside'])
def test_unsafe_relative_paths_rejected(value):
    with pytest.raises(AssertionError): package.safe_relative(value)


def test_routing_rejects_duplicate_population_and_other_experiments():
    good=routing();package.routed_attempts(good)
    bad=copy.deepcopy(good);bad['workers'][1]['condition']='gc100'
    with pytest.raises(AssertionError):package.routed_attempts(bad)
    bad=copy.deepcopy(good);bad['workers'][0]['name']='unrelated-worker'
    with pytest.raises(AssertionError):package.routed_attempts(bad)


def test_cleanup_requires_both_resources_and_all_attempts(tmp_path):
    route=routing();proof=cleanup_fixture(tmp_path,route)
    assert len(package.verify_cleanup(tmp_path,route,proof))==4
    bad=copy.deepcopy(proof);bad['workers'][0]['disk_absent']=False
    with pytest.raises(AssertionError):package.verify_cleanup(tmp_path,route,bad)
    bad=copy.deepcopy(proof);bad['workers'][0]['evidence'].pop()
    with pytest.raises(AssertionError):package.verify_cleanup(tmp_path,route,bad)
    prior=copy.deepcopy(route['workers'][0]);prior['name']=prior['name'].replace('-01','-00')
    prior['local_artifact_root']=prior['local_artifact_root'].replace('-01','-00')
    route['prior_attempts']=[prior]
    with pytest.raises(AssertionError):package.verify_cleanup(tmp_path,route,proof)
    expanded=cleanup_fixture(tmp_path,route)
    assert len(package.verify_cleanup(tmp_path,route,expanded))==6


def test_cleanup_rejects_changed_evidence(tmp_path):
    route=routing();proof=cleanup_fixture(tmp_path,route)
    (tmp_path/proof['workers'][0]['evidence'][0]['path']).write_text('different log')
    with pytest.raises(AssertionError):package.verify_cleanup(tmp_path,route,proof)


def test_full_tree_retains_failed_and_ancestor_evidence(tmp_path):
    folder=tmp_path/'experiment';folder.mkdir()
    names=['final.pt','failed_plan.pt','incomplete.json',
           'resume_parent_evidence/ancestor_evidence/metrics.jsonl']
    for name in names:
        path=folder/name;path.parent.mkdir(parents=True, exist_ok=True);path.write_bytes(b'{}')
    found=package.experiment_files(tmp_path,'experiment')
    assert {str(p.relative_to(folder)) for p in found}==set(names)
    (folder/'credentials.json').write_text('{}')
    with pytest.raises(AssertionError):package.experiment_files(tmp_path,'experiment')


def test_symlink_and_private_material_rejected(tmp_path):
    outside=tmp_path/'actual';outside.mkdir();(outside/'data.json').write_text('{}')
    (tmp_path/'link').symlink_to(outside, target_is_directory=True)
    with pytest.raises(AssertionError):package.checked_path(tmp_path,'link/data.json')
    p=tmp_path/'result.json';p.write_text(json.dumps(dict(access_token='synthetic-private-value')))
    with pytest.raises(AssertionError):package.inspect_file(p)


def test_partial_audit_refused_before_any_file_reads(tmp_path):
    for report in (dict(passed=True,complete=False,status='pending'),
                   dict(passed=False,complete=True,status='failed')):
        with pytest.raises(AssertionError):package.verify_final_audit(tmp_path,tmp_path/'missing','gc100',report)


def test_complete_label_cannot_hide_a_missing_stage(tmp_path):
    source=tmp_path/'scripts/audit_local_goal_return_training.py'
    source.parent.mkdir();source.write_text('# synthetic audit source\n')
    folder=tmp_path/'results/experiment';folder.mkdir(parents=True)
    report=dict(passed=True,complete=True,status='complete',failures=[],pending=[],condition='gc100',
        protocol_sha256=package.PROTOCOL_SHA,folder=str(folder),auditor_sha256=package.sha(source),
        audit_dependency_sha256={},checks=dict(final_exact_stage4=True,final_summary=True),artifacts={})
    for stage in range(1,4):
        report['checks'].update({f'stage{stage}_evaluation_preserved_runtime':True,
            f'stage{stage}_raw_evaluation':{},f'stage{stage}_before_evaluation.pt':{},f'stage{stage}.pt':{}})
    with pytest.raises(KeyError,match='stage4'):
        package.verify_final_audit(tmp_path,folder,'gc100',report)


def test_archive_is_deterministic_and_checks_every_byte(tmp_path):
    (tmp_path/'a.json').write_text('{"synthetic":true}\n')
    (tmp_path/'b.pt').write_bytes(b'fake checkpoint bytes for archive testing only')
    entries=[dict(path=name,bytes=(tmp_path/name).stat().st_size,sha256=package.sha(tmp_path/name))
             for name in ('b.pt','a.json')]
    package.write_archive(tmp_path,tmp_path/'one.tar.gz',entries)
    package.write_archive(tmp_path,tmp_path/'two.tar.gz',entries)
    assert package.sha(tmp_path/'one.tar.gz')==package.sha(tmp_path/'two.tar.gz')
    with tarfile.open(tmp_path/'one.tar.gz') as archive:
        assert archive.getnames()==['a.json','b.pt']
        assert all(m.uid==0 and m.gid==0 and m.mtime==0 and m.isfile() for m in archive)
    (tmp_path/'a.json').write_text('changed')
    with pytest.raises(AssertionError):package.write_archive(tmp_path,tmp_path/'bad.tar.gz',entries)
