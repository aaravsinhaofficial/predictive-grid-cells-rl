# Bounded compute for the ring-attractor study

Audit date: 5 November 2026. The audit used read-only GCP commands. It created no cloud resources. The separate launcher below is used only when the root agent explicitly runs `launch`.

## Recommendation and verified access

Start with one on-demand `g2-standard-4` worker for a two-hour pilot. It has one L4 GPU and is sufficient for small recurrent models; use the measured training throughput to decide the later allocation. `g2-standard-8` is available if CPU work or memory limits the first machine. Avoid a large fleet before validating the scientific code and measuring throughput.

For the small controlled RNN, use `e2-standard-8` when local timing shows CPU throughput is adequate or L4 capacity is unavailable. It has eight vCPUs and 32 GiB RAM; its audited Iowa on-demand price is **$0.26804568/hour**. The same $1.25/hour budget reservation remains in force. CPU startup skips GPU installation and CUDA checks; the experiment command must explicitly use `--device cpu`. [Official general-purpose pricing](https://cloud.google.com/products/compute/pricing/general-purpose).

The active project is `connectome-ai-research`. In `us-central1`, the read-only quota response showed 256 L4 GPUs allowed and none used, plus 128 T4 GPUs allowed and one used. In `us-east1`, it showed 16 L4 GPUs and eight T4 GPUs allowed, all unused. These are quotas, not promises of immediate physical capacity.

The accelerator list offered L4 in `us-central1-a/b/c`, `us-east1-b/c/d`, and `us-west1-a/b/c`. The launcher deliberately restricts deployment to the price-audited `us-central1-a/b/c` zones.

At audit time, the project had 16 terminated `mech-a11-*` VMs and one unrelated running VM named `instance-20261008-004847`. Leave all of them untouched. The new launcher only handles exact names recorded in its own ledger and verifies their study labels and instance IDs before deletion.

## Prices and budget

Official Iowa prices checked on the audit date:

| Machine | On demand, USD/hour | Current Spot, USD/hour |
| --- | ---: | ---: |
| `g2-standard-4`, one L4 | 0.706832276 | 0.423016 |
| `g2-standard-8`, one L4 | 0.853624312 | 0.510832 |

These machine prices include the GPU. Spot pricing can change; the pilot uses on-demand capacity. The official T4 GPU-only rate is $0.35/hour, before adding its host machine. [Official accelerator pricing](https://cloud.google.com/products/compute/pricing/accelerator-optimized).

A 100 GiB balanced boot disk adds approximately $0.0137/hour in Iowa. It is configured to delete with the VM. [Official disk pricing](https://cloud.google.com/compute/disks-image-pricing).

`ring_cloud_budget.json` reserves **$1.25 per VM-hour**, including a **15-minute allowance per launch**, and holds **$25** separately for ancillary storage/transfer charges. A two-hour pilot reserves $2.8125, making the initial committed envelope $27.8125 including the ancillary reserve. This is conservative accounting, not a claim about the final invoice. Every creation reserves its complete envelope first. Creation is refused if closed cost bounds plus active reservations plus the ancillary reserve would exceed $500.

There is one active worker slot until a successful pilot has been collected and deleted. Only then may `set-parallelism` increase the limit, to at most eight. Every worker independently retains the full reservation until its absence is verified. No restart or replacement of an old worker is automatic.

## Isolation, image, and output

The existing bucket `connectome-ai-research-popcode` is Standard storage in `US-CENTRAL1`, with seven-day soft deletion. The launcher uses a new prefix:

`gs://connectome-ai-research-popcode/ring-20261105/<unique-vm-name>/`

The image is pinned to `pytorch-2-9-cu129-ubuntu-2204-nvidia-580-v20261029` in `deeplearning-platform-release`. It was READY when checked, with a 100 GiB minimum disk. Pinning avoids silently changing the Python/CUDA environment between workers.

Only source files under `code`, `scripts`, and `tests` are packaged by default. Additional input files, including checkpoints, require explicit `--include FILE`. Each source is hashed and the same checked bytes are archived; a concurrent edit aborts packaging. The VM checks the complete archive hash. Packages are limited to 2 GiB and output trees to 5 GiB. The worker records its command, completion status, and source-manifest hash.

VM access scopes include storage read/write, without compute write permission. The worker uploads its isolated result directory every five minutes and writes `DONE.json` after the final upload. The root agent collects and deletes completed workers promptly. A worker remains running after completion so its cloud-enforced deletion deadline continues to apply; it does not issue a guest shutdown that could strand a stopped disk.

## Independent runtime limit

Every launch supplies `--max-run-duration`, `--instance-termination-action=DELETE`, `--no-restart-on-failure`, and boot-disk automatic deletion. The root agent verifies the returned scheduling configuration. This provider-side limit does not depend on the local laptop staying awake. Google documents a possible additional 30 seconds before automatic termination begins. [Runtime-limit documentation](https://docs.cloud.google.com/compute/docs/instances/limit-vm-runtime).

The worker stops its experiment before the deadline to leave time for final upload. A failed command records its exit code. Startup failure is visible in `logs`; a missing `DONE.json` never means scientific success.

## Commands

```sh
# Review the package and cost envelope without creating resources.
python scripts/ring_cloud.py plan --name ring-20261105-pilot-01 \
  --command 'python scripts/EXPERIMENT.py --manifest configs/PILOT.json' \
  --include configs/PILOT.json --output-dir results/ring_attractor --hours 2

# Root executes the reviewed plan by replacing "plan" with "launch".
python scripts/ring_cloud.py status
python scripts/ring_cloud.py logs ring-20261105-pilot-01
python scripts/ring_cloud.py collect ring-20261105-pilot-01
python scripts/ring_cloud.py delete ring-20261105-pilot-01
python scripts/ring_cloud.py ledger
```

`delete` collects first by default. `--without-collect` is reserved for failed workers with no artifacts. Collected files go to `results/ring_cloud_artifacts/<vm-name>/`. After the study, verify local retention and remove only the study's cloud objects; the existing bucket and unrelated objects remain intact. Keep the ledger and artifact manifests with the research record.

The older `scripts/gcp_fleet.py` should not be used unchanged: it lacks enforced VM deadlines and uses a hardcoded Spot discount that does not match the audited prices.

Validation: seven offline safety tests in `tests/test_ring_cloud.py` pass. These cover budget refusal before cloud mutation, pilot concurrency, path/name isolation, replacement-instance protection, stable closed-cost accounting, compilation of the embedded worker, and CPU startup without a GPU dependency. No cloud launch was performed by the audit agent.

## Capacity fallback update

L4 launches in us-central1-a/b/c returned stockout and created no VMs. N1-standard-8 plus one T4 is also permitted: official Cloud Billing Catalog SKUs on2026-11-05 list8cores×$0.031611/h +30GiB×$0.004237/h +T4$0.35/h = **$0.729998/h**, below the existing$1.25/h reservation. The same TTL, artifact and identity guards apply. CPU E2 uses MIGRATE maintenance; attached-GPU VMs use TERMINATE.

Scaling may proceed after a complete training job within the multi-job pilot has been collected and its checkpoint hashed, even while the remaining pilot jobs continue. The ledger records that verification. Budget reservations and provider deletion deadlines still apply independently to every VM.
