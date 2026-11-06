# Secondary navigation outcomes fixed before confirmation evaluation

Specified locally on 5 November 2026, after inspecting pilot navigation but before computing these outcomes on confirmation seeds. These are secondary outcomes, separate from the four locked primary heading/anchoring comparisons.

The controller and original arrival, success, distance and path-length measures stay unchanged. Report the following at both 20 and 40 seconds:

- **Post-entry excursion:** maximum true distance from the goal after the first entry into its 15 cm radius. Also report the amount beyond the radius. These summaries are conditional on entering, and must include the number of entrants. Nonentrants have missing post-entry values, never an artificial zero.
- **Terminal occupancy:** fraction of the final five seconds spent within the 15 cm goal radius, averaged over every trial, including failures.
- **Sustained dwell:** two consecutive seconds inside the goal with forward speed at or below 5 cm/s. Report the fraction achieving this, the time to first achievement capped at the episode deadline, and the longest qualifying run. All trials remain in these summaries.

These measures distinguish merely passing through a goal from maintaining an accurate stopped position. They are calculated by the scorer and never feed back into the agent.

Interpret reversal mechanistically only for a PGC-trained network: its existing predictive displacement is reversed. For Grid and Grid+explicit-heading networks, the evaluator's reversed branch introduces unfamiliar shifted spatial inputs; it is an out-of-distribution diagnostic, not a selective reversal of an input those networks learned to use. The separate anchoring evaluator uses no-op reversal controls for those populations, so the two reports' non-PGC reversal branches must not be conflated.
