# PySpot

New version of this [repository](https://github.com/saigrain/pyspot) for simulating the lightcurves of stars with starspots.

## Usage
Once installed PySpot can be run by calling `pyspot pyspot.yaml` where `pyspot.yaml` is the included parameter file. It will then generate a set of star spot parameters and produce the corresponding lightcurve using the equations published in [Kipping (2012)](https://ui.adsabs.harvard.edu/abs/2012MNRAS.427.2487K/abstract). Running `pyspot --help` describes additional options controlling the output.

## Changes
 - Implemented the [Kipping (2012)](https://ui.adsabs.harvard.edu/abs/2012MNRAS.427.2487K/abstract) lightcurve model. This adds, among other things, limb-darkening of the star.
 - Changed to size evolution of the star spots to a squared-exponential profile, eliminating a discontinuity in the size profile.
 - Added more control over the stars activity level and the phase of the activity cycle, making it easier to simulate stars at peak activity.
 - Various bug fixes.

