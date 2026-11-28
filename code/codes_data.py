"""Recorded numbers used to build population codes.

GARDNER_PGC_SHIFT_CM: best forward shift (cm) of every predictive grid cell in the lab's
reanalysis of the Gardner et al. (2022) Neuropixels recordings (one entry per module
recording: rat Q modules 1-2, rat R day 1 and day 2 modules 1-3, rat S module 1). A cell
counts as predictive when its grid score peaks at a positive spatial shift (position
projected ahead along the direction of movement, 2 cm steps between -50 and 50 cm), the
peak beats a shuffle and exceeds 0.3, and the grid score at zero shift is below 0.5.
GARDNER_MODULE_SPACING_CM: mean grid spacing of each module (analysis notebooks of Redman
et al. 2025; the day-2 recordings of rat R use the day-1 values of the same modules).
GARDNER_SHIFT_FRAC: every shift divided by its module spacing (108 cells; mean 0.097,
median 0.069).
"""
GARDNER_MODULE_SPACING_CM = {
    'q1': 67.6,
    'q2': 101.1,
    'r11': 57.6,
    'r12': 87.8,
    'r13': 113.6,
    'r21': 57.6,
    'r22': 87.8,
    'r23': 113.6,
    's1': 55.0,
}
GARDNER_PGC_SHIFT_CM = {
    'q1': [2, 2, 4, 4, 8, 12, 12, 12],
    'q2': [2, 2, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6],
    'r11': [2, 2, 4, 4, 8, 10, 10, 10, 12, 12],
    'r12': [2, 2, 2, 2, 2, 2, 4, 4, 6, 8, 8, 8, 10, 10, 16, 22],
    'r13': [2, 2, 4, 4, 6, 6, 6, 8, 10, 10, 14, 20, 50],
    'r21': [2, 2, 2, 4, 6, 8, 8, 8, 8, 8, 8, 8, 10, 10, 12, 16],
    'r22': [2, 2, 2, 4, 4, 6, 6, 8, 8, 8, 8, 8, 12, 12, 14, 14],
    'r23': [2, 2, 4, 4, 4, 6, 6, 6, 6, 10, 10, 14, 18, 28, 48, 50],
    's1': [2],
}
GARDNER_SHIFT_FRAC = tuple(round(s / GARDNER_MODULE_SPACING_CM[m], 4)
                           for m, v in GARDNER_PGC_SHIFT_CM.items() for s in v)
GARDNER_SHIFT_M = tuple(round(s / 100.0, 4) for v in GARDNER_PGC_SHIFT_CM.values() for s in v)
