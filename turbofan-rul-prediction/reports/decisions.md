* Identified constant columns (s1, s5, s10, s16, s18, s19, op3) via `nunique()==1`; std-based check wasunreliable due to floating-point noise in s5 and s16.
* op1 and op2 vary only by about ±0.0087 and ±0.0006 with correlation near 0 against RUL, so they are treated as jitter around the single operating condition and excluded.
* **s6** is two-valued with weak correlation (-0.13)
* **s9** and **s14** kept despite moderate correlation because the plots show clear, engine-dependent drift. 
