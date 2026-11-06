# A physical bound for the observed two-action repertoire

This secondary question was specified after observing retained-actor baseline action counts, before calculating this bound. It does not change training, endpoints, criteria or the later memory experiment. All 16 sealed contexts and 16 headings are retained; only immutable **pretraining** baseline snapshots are read.

Native actions 1 and 5 are clockwise turning and rightward strafing. Let `q = −rotation velocity`, `r = −leftward velocity`, `a = exp(−dt/0.25)`, rotation limit `w`, and translation limit `v`. Separate additive motor noise from control. Initially translation is zero and rotation is LEFT. The controlled motors satisfy `q_t = a q_(t−1) + (1−a) w I(action1)` and `r_t = a r_(t−1) + (1−a) v I(action5)`. Consequently `z_t = q_t/w + r_t/v` has the action-independent recurrence `z_t = a z_(t−1) + 1−a`.

Define controlled distance `L_t = dt Σr` and clockwise angle `Q_t = dt Σq`. Then `Q_t = w(Z_t − L_t/v)`, where `Z_t = dt Σz`. Since q starts nonpositive and has nonnegative targets, it can cross zero at most once. Therefore the largest Q before a hit is at the start or the hit. A hit requires `L_hit ≥ D−R−E`, with goal-center distance D, goal radius R and translation-noise bound `E = dt Σ||filtered body noise||`. This bounds the maximum Q using `max_prefix Z`; decaying initial LEFT momentum bounds its minimum.

At each update the controlled strafe vector points along **updated heading minus π/2**. Add the maximum cumulative angular-noise excursion to the angle range. If this sector is narrower than π, its nonnegative vector sums stay inside it. A goal can then be reached only when its bearing lies inside the sector expanded by `asin((R+E)/D)`. Require `D>R+E` and certify the entire possible path wall-free using `v×duration+E` versus wall clearance. Failed premises remain potentially reachable.

Fixed margins of 0.1 mm and 0.0001 rad cover ordinary numerical sensitivity; this is an analytical native-equation bound, not a formal verification of hardware floating-point instructions. Exhaustive synthetic short sequences check the signs and preterminal-hit logic.

The result bounds **any controller restricted to actions {1,5}**, including adaptive switching. It does not bound the six-action networks: finite softmax probabilities can include every action, and the PGC baseline also sampled nine backward actions. A passing angle condition is necessary, not a constructive navigation solution. The calculation diagnoses a possible action-repertoire bottleneck, not memory quality or ring geometry.
