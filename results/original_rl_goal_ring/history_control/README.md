# Grid history can provide the same arrival cue

We ran the original frozen Grid actor in the same circular-goal square arena. At every first pickup, we saved up to 30 recent ordinary-grid observations and the actual pickup snapshot. A known-map inverse recovered their positions. The last displacement exceeding the original movement threshold supplied arrival direction; smaller movements retained that direction. No true direction was supplied to this reconstruction.

Included **64 first pickups from 64 episodes**. History supplied a direction for **64/64**; missing histories were not filled using ground truth.

The resulting ahead point differed from the independently decoded PGC ahead point by at most **9.54e−8 m**. Ring phase differed by at most **5.12e−6°**. Regenerated 512-cell goal-code activity differed by at most **1.13e−6**.

**Meaning:** the useful arrival cue can also be reconstructed from ordinary spatial history. Predictive cells supply it directly in the saved goal message. This control shows information availability; it does not claim that the original Grid LSTM actually performs the explicit inverse and displacement calculation.

The current ArenaEnv implementation updates movement direction from the whole decision-step displacement, despite `action_repeat=4`. This is why the observed-position history matches its direction convention. The last-moving threshold is 0.05 m/s × 4/60 s. The previous 30 observations are an explicit finite history window; every first pickup and any unavailable direction are retained in [results.json](results.json).

No new behavioral cohort was needed to compare identical ring outputs. The actor was unchanged, and true positions, directions, and goals were used only for diagnostics.
