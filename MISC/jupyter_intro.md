# Jupyter Lab and Jupyter Notebooks (R and Python)

## Install Conda Python

* [Miniconda](https://docs.conda.io/en/latest/miniconda.html) - My preferred install.  Just contains the minimum needed to run Python, other packages can be installed as needed.
* [Anaconda](https://www.anaconda.com/distribution/) - Contains all packages that a data scientist might need in Python, very large install.

## Python and R Virtual Environments

Python has [three](https://medium.com/@krishnaregmi/pipenv-vs-virtualenv-vs-conda-environment-3dde3f6869ed) virtual environment managers.

* [Virtualenv](https://docs.python.org/3/tutorial/venv.html) - The origional, though mostly replaced Python environment manager. Still a very good option for Python 2.7.
* [PipEnv](https://github.com/pypa/pipenv) - Addresses issues in Virtualenv, such as multi-level dependancies. 
* [Conda](https://www.heatonresearch.com/2018/01/01/python-care-feeding.html) - My preferred environment manager. 

## Popular Editors and IDEs for Python

* [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/)
* [Jupyter Hub](https://jupyter.org/)
* [VSCode](https://code.visualstudio.com/)
* [Atom](https://atom.io/)
* [PyCharm](https://code.visualstudio.com/)
* [Spider](https://www.spyder-ide.org/)

## Install Jupyter Notebook and Jupyter Lab

```
conda install jupyter
conda install -c conda-forge jupyterlab
```

## Create a Python Virtual Environment

* [Care and Feeding of Python Environments](https://www.heatonresearch.com/2018/01/01/python-care-feeding.html)
* [T81-558:Applications of Deep Neural Networks](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_01_1_overview.ipynb)

## My Python Environment (WUSTL)

*Note: change second line to ```activate tensorflow``` for Windows.

```
conda create -y --name tensorflow python=3.6
source activate tensorflow
conda install -y jupyter
conda install -y scipy
pip install --exists-action i --upgrade sklearn
pip install --exists-action i --upgrade pandas
pip install --exists-action i --upgrade pandas-datareader
pip install --exists-action i --upgrade matplotlib
pip install --exists-action i --upgrade pillow
pip install --exists-action i --upgrade tqdm
pip install --exists-action i --upgrade requests
pip install --exists-action i --upgrade h5py
pip install --exists-action i --upgrade pyyaml
pip install --exists-action i --upgrade tensorflow_hub
pip install --exists-action i --upgrade bayesian-optimization
pip install --exists-action i --upgrade spacy
pip install --exists-action i --upgrade gensim
pip install --exists-action i --upgrade flask
pip install --exists-action i --upgrade gym
pip install --exists-action i --upgrade tf-nightly-2.0-preview
pip install --exists-action i --upgrade keras-rl2 --user
python -m ipykernel install --user --name tensorflow --display-name "Python 3.6 (tensorflow)"
conda update -y --all
```

## My R Environment

```
conda create -n r_env r-essentials r-base
source activate r_env
conda install -y r-caret
conda install -y r-pRoc
conda install -y r-doParallel
conda install -c conda-forge r-dosnow
conda install -y r-data.table
conda install -y r-lubridate
conda install -y r-survival
conda install -y r-ranger
conda install -y r-plyr
conda install -y r-dplyr
conda install -y r-car
conda install -y r-tensorflow
conda install -y r-keras
conda install -y r-reticulate
conda install -y r-rms
conda install -y r-ggplot2
conda install -y r-openxlsx
conda install -y r-splines
```

## Jupyter Lab Extensions

* [Installing Jupyter Lab Extensions](https://jupyterlab.readthedocs.io/en/stable/user/extensions.html) - Use the extension manager
* [A curated list of awesome JupyterLab extensions and resources](https://github.com/mauhai/awesome-jupyterlab)