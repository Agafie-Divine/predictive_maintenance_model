# Turbofan Remaining Useful Life (RUL) Prediction

Predicting how many operating cycles a turbofan engine has left before failure, using NASA's C-MAPSS simulated run-to-failure data, and connecting the model's behaviour to the mechanical degradation of the high-pressure comprssor (modeled in solidworks)

## Problem
### What is being predicted?
For each jet engine, the model predicts its **Remaining Useful Life (RUL)**: the number of operating cycles left before the engine fails, given the sensor history observed up to its latest cycle. This is a regression problem, since the target is a number of cycles.

### From what data?
The project uses subset FD001 of NASA's **C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation) turbofan dataset. It is a simulated data.
-**Training set:** 100 engines, each recorded from healthy operation until failure (20,631 rows), so the true RUL at every cycle can be derived.
-**Test set:** 100 engines whose recordings stop *before* failure (13,096 rows). A separate file gives the true **RUL** at each engine's last observed cycle, which is what the model is evaluated against.
-**Features per cycle:** 3 operational settings and 21 sensor measurements.
-**FD001 conditions:** one operating condition and one fault mode (degradation of the high-pressure compressor). Engines have different lifetimes, so the model must learn degradation pattern.

### Why does this matter for maintenance planning?
Maintenance strategies are usually one of the three kinds:
-**Run to failure:** This is cheap until unplanned failures which then becomes costly and, in aviation, a safety risk.
-**Fixed schedule:** Parts are replaced after a set number of cycles. This is safe but wasteful, because engines can be pulled with useful life left.
-**Condition-based / predictive:** Action is taken when the engine's actual condition says it is needed.

An accurate RUL estimate enables the third approach. Maintenance teams can order parts, schedule technicians and book hangar time in advance, and intervene before failure without retiring healthy engines early. The costs are also **asymmetric**: predicting failure too late (an unplanned failure) is generally worse than predicting it too early (some wasted life). That is why this project is evaluated with both RMSE and NASA's asymetric scoring function.
