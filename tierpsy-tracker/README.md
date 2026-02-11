# Tierpsy Tracker (install Python)
This is a light-weight (not tested if the GUI works!) fork of the [Tierpsy Tracker][1].  
Some changes were made to be able to easily install and use it as a module (in python script/jupyter/...).  
And don't forget to cite their work [Tierpsy Tracker][1]! :)

Follow these steps to install and use it.

## Installation
0. You need to install [conda][2] first. (All you need to be able to do is to create/manage conda environments) and also [Microsoft C++ Build Tools][3].
1. Clone the repo in the `development` brance
```bash
git clone https://github.com/SinRas/tierpsy-tracker
```
2. Create conda environment with python=3.8, e.g. run following script (which creates a conda environment named `tt`)
```bash
conda create -n tt python=3.8
```
and activate it
```bash
conda activate tt
```
3. Install requirements (inside conda environment `tt`) using pip, e.g. by running scrip
```bash
pip install -r requirements.txt
```
4. Install `tierpsy` module from the cloned folder/repository using:
```bash
cd tierpsy-tracker  # if you have not changed directory to this repository in command line
pip install -e .
```
5. Test if you can import the module inside your scripts/jupyter notebooks (make sure the environment `tt` is being used for running your script/jupyter lab), e.g. run following lines inside your code
```python
import tierpsy
from tierpsy.features import tierpsy_features
from tierpsy.analysis.ske_create.getSkeletonsTables import getWormMask, getSkeleton
```



[1]: https://github.com/Tierpsy/tierpsy-tracker/tree/development
[2]: https://conda.io/projects/conda/en/latest/user-guide/install/index.html
[3]: https://visualstudio.microsoft.com/visual-cpp-build-tools/
