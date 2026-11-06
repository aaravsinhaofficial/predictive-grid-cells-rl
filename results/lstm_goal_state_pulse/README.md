# Why a linear decoder edit was not a useful goal-memory intervention

We used the original seed1 PGC+grid reinforcement-learning agent. Its policy weights, square arena and real actor were unchanged. Separate native episodes trained a linear diagnostic readout of goal bearing from the LSTM cell state and an independent nonlinear checker. Neither readout supplied actions or observations to the agent.

At26 fresh return states, a constrained cell-state edit rotated the **linear readout by15°**, while preserving its linear position, heading and goal-distance readouts. The actor-facing hidden state stayed fixed. Equal-norm random and decoder-silent edits were controls.

The independent nonlinear checker shifted only **1.2°** on average. Both readouts had substantial absolute error on fresh states:32.7° linear and36.9° nonlinear. With the same real observations replayed, the nonlinear displacement fell to0.055° within five decisions. Actual first-second path change averaged0.56mm; goal hits remained25/26. Random and decoder-silent edits generally changed action probabilities more.

The nominal15° edit therefore did not behave like a reliable15° change in the agent's goal belief. It is retained as a failed diagnostic, not evidence for an attractor or predictive repair. A stronger follow-up elicits alternative states through the agent's **actual saved goal-code input** and then restores the real input: [natural goal-state intervention](../lstm_goal_message_state_pulse/README.md).

Full methods, independent validation and all controls are in [the pilot summary](pilot_pgc33_s1/summary.json). The c-only manipulation deliberately held h fixed; about14.7% of coordinates then violated the instantaneous relation `h=o*tanh(c)`, another reason not to describe the edited state as naturally occupied.
