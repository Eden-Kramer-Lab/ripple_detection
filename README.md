# ripple_detection
`ripple_detection` is a python package for finding sharp-wave ripple events (150-250 Hz) from local field potentials.

The package implements ripple detection methods from Karlsson et al. 2009 (`Karlsson_ripple_detector`) and Kay et al. 2016 (`Kay_ripple_detector`).

### Installation ###
```python
pip install ripple_detection
```
OR
```python
conda install -c edeno ripple_detection
```

### Package Dependences ###
+ numpy
+ scipy
+ pandas

### Example Useage ###
```python
from ripple_detection import Kay_ripple_detector

ripple_times = Kay_ripple_detector(
  time, LFPs, speed, sampling_frequency)
```

### References ###
1. Karlsson, M.P., and Frank, L.M. (2009). Awake replay of remote experiences in the hippocampus. Nature Neuroscience 12, 913-918.

2. Kay, K., Sosa, M., Chung, J.E., Karlsson, M.P., Larkin, M.C., and Frank, L.M. (2016). A hippocampal network for spatial coding during immobility and sleep. Nature 531, 185-190.
