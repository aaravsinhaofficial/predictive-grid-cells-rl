"""Independent arithmetic crosscheck of the completed fixed90M actor readout.

Uses minimum-norm least squares for the actor rowspace and SciPy softmax;
does not call the original decompose helper or recompute recurrent histories.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import torch
from scipy.special import softmax
from threadpoolctl import threadpool_limits
import lstm_reorientation_entropy_actor_readout as producer
from lstm_endogenous_curve_recovery import sha


def run():
    folder = producer.OUT
    summary = json.loads((folder / 'summary.json').read_text())
    protocol = json.loads((folder / 'protocol.json').read_text())
    assert protocol == producer.protocol()
    assert summary['protocol_sha256'] == sha(folder / 'protocol.json')
    assert summary['complete'] and summary['all128rows_retained']
    assert len(summary['rows']) == 128
    for path, digest in summary['source_sha256'].items():
        assert sha(ROOT / path) == digest
    indexed = {(r['arm'], r['condition'], r['context'], r['cue'], r['time_updates']): r
               for r in summary['rows']}
    assert len(indexed) == 128
    errors = {}; checked = 0; native = []

    def compare(key, actual, expected):
        error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
        errors[key] = max(errors.get(key, 0), error)
        assert np.allclose(actual, expected, atol=2e-10, rtol=2e-10), (key, error)

    for arm, condition in producer.MODELS:
        checkpoint = producer.producer.resolve_checkpoint(condition, arm)
        data, cfg = producer.producer.preflight(condition, arm, checkpoint, producer.producer.ROUTING)
        digest = sha(checkpoint)
        w = data['policy']['pi.weight'].double().numpy()
        b = data['policy']['pi.bias'].double().numpy()
        a = w - w.mean(axis=0)
        paths = producer.INPUT / arm / condition
        for context in (0, 1):
            path = paths / f'context{context}_states.npz'
            archive_hash = sha(path)
            with np.load(path) as loaded:
                archive = {k: loaded[k] for k in loaded.files}
            producer.validate_inputs(data, condition, arm, archive, digest, paths)
            for cue in producer.original.CUES:
                for time in producer.original.TIMES:
                    row = indexed[(arm, condition, context, cue, time)]
                    assert row['checkpoint_sha256'] == digest and row['archive_sha256'] == archive_hash
                    assert row['genuine_extra_frames'] == 90_132_480
                    h = archive[cue + '_states'][list(archive['times']).index(time), :, :256].astype(np.float64)
                    logits = h @ w.T + b
                    qlogits = logits - logits.mean(axis=1, keepdims=True)
                    mean = qlogits.mean(axis=0)
                    signal = qlogits - mean
                    dh = h - h.mean(axis=0)
                    # Solve A x = each centered logit variation for the smallest
                    # hidden-state vector, rather than constructing an SVD basis.
                    projected = np.linalg.lstsq(a, signal.T, rcond=max(a.shape)*np.finfo(float).eps)[0].T
                    energy = np.mean(np.sum(dh*dh, axis=1))
                    row_fraction = np.mean(np.sum(projected*projected, axis=1)) / energy
                    null_fraction = np.mean(np.sum((dh-projected)**2, axis=1)) / energy
                    p = softmax(logits, axis=1)
                    vector_rms = lambda x: np.sqrt(np.mean(np.sum(np.asarray(x)**2, axis=1)))
                    compare('centered_logit_mean', mean, row['centered_logit_mean'])
                    compare('centered_logit_vector_rms', vector_rms(signal), row['centered_logit_family_rms'])
                    compare('rowspace_fraction', row_fraction, row['actor_rowspace_hidden_variance_fraction'])
                    compare('nullspace_fraction', null_fraction, row['actor_nullspace_hidden_variance_fraction'])
                    compare('energy_partition', row_fraction + null_fraction, 1)
                    compare('probability_mean', p.mean(axis=0), row['probabilities']['mean_across_phases'])
                    compare('probability_vector_rms', vector_rms(p-p.mean(axis=0)), row['probabilities']['family_rms'])
                    compare('translation_probability', p[:, 2:].sum(axis=1).mean(), row['probabilities']['total_translation_mean'])
                    compare('right_vs_translation_margin', mean[1]-mean[2:].max(), row['right_turn_minus_best_translation_mean_logit'])
                    compare('softmax_common_score_invariance', softmax(qlogits, axis=1), p)
                    if cue == 'native' and time == 60:
                        native.append(dict(arm=arm, condition=condition, context=context,
                            centered_logit_vector_rms=vector_rms(signal),
                            rowspace_fraction=row_fraction,
                            right_minus_best_translation_mean_logit=float(mean[1]-mean[2:].max()),
                            total_translation_probability=float(p[:, 2:].sum(axis=1).mean()),
                            actor_probability_vector_rms=vector_rms(p-p.mean(axis=0))))
                    checked += 1
            assert sha(path) == summary['source_sha256'][str(path.relative_to(ROOT))]
        assert sha(checkpoint) == digest
    assert checked == 128
    return dict(passed=True, all128rows_verified=True, checked_rows=checked,
        protocol_sha256=sha(folder / 'protocol.json'), numerical_source_sha256=sha(producer.__file__),
        summary_sha256=sha(folder / 'summary.json'), auditor_sha256=sha(__file__),
        method='Independent direct centered logits, minimum-norm least-squares actor-rowspace projection, SciPy softmax, energy partition and mean action margins. No call to decompose; no recurrent, environment or training update.',
        max_abs_errors=errors, native_at4seconds=native,
        scope='All four fixed90M endpoints; one original seed pair. RMS values use vector norms, not per-action RMS. Native contexts differ between arms; descriptive contrasts are not paired controlled hidden states.')


if __name__ == '__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        output = run()
    (producer.OUT / 'completed_output_audit.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({k: output[k] for k in ('passed', 'checked_rows', 'max_abs_errors')}))
