# Sequential Search Transformer (SST)

This repository contains the code implementation for the Monte Carlo simulations conducted to evaluate parameter recovery capabilities of the **Sequential Search Transformer (SST)**, as described in our recent study.

## Overview

Understanding and leveraging consumer behavior presents significant business opportunities. Traditional deep learning methods, despite their predictive strengths, lack interpretability and explicit modeling of consumer decision-making processes. Economic theories suggest that consumers typically follow a sequential search strategy, evaluating alternatives sequentially to find the best match for their preferences.

To address this challenge, we developed the **Sequential Search Transformer (SST)**, which integrates deep learning approaches with economic sequential search theory to accurately model consumer search and purchase decisions.

## Key Contributions

- SST combines the predictive power of deep learning with economic theory, resulting in improved interpretability and decision modeling.
- SST explicitly models consumer search and decision-making across sessions and sequentially resolves uncertainty in product utility.
- Demonstrated through Monte Carlo simulations, SST effectively recovers parameters from simulated datasets.
- Empirical evaluations indicate SST's superior performance compared to state-of-the-art deep learning and structural econometric models.

## Monte Carlo Simulations

The simulations provided in this repository serve the following purposes:

- **Parameter Recovery:** Verify SST's capability to recover known parameters from simulated datasets.
- **Model Validation:** Assess the robustness and reliability of SST under controlled conditions.


## Repository Structure

```bash
├── VAR=0.1/                    # Simulated data (simulated_sessions.7z, need to unzip the file), trained model parameters (DeepStructural_model_final.pt), and logs with variance = 0.1
├── VAR=0.5/                    # Simulated data (simulated_sessions.7z, need to unzip the file), trained model parameters, and logs with variance = 0.5
├── VAR=1.0/                    # Simulated data (simulated_sessions.7z, need to unzip the file), trained parameters, and logs with variance = 1.0
├── ru_model.py                 # Reservation Utility model
├── ru_model_parameters.pt      # Pre-trained Reservation Utility model parameters
├── simulated_sessions.pkl      # Generated simulated session data
├── Simulated_Date_PrePost_Shock.py   # Script to generate simulated session data
├── deep_structural_embedding_prepost_shock.py  # SST training script for real-world or simulated data
└── README.md
```



## Usage


### Using Pre-trained Model and Simulated Data

To evaluate SST using pre-simulated data and trained parameters:

1. Copy the files `simulated_sessions.pkl` and `DeepStructural_model_final.pt` from their respective subdirectories.
2. Run:
   ```bash
   python deep_structural_embedding_prepost_shock.py
   ```

This will load the simulated data and trained parameters to test the parameter recovery performance.

### Full Simulation and Training Process

To run the complete simulation and training process:

1. Execute the simulation script to generate data:
   ```bash
   python Simulated_Date_PrePost_Shock.py
   ```
   *(Adjust the preference shock variance parameter within the script at line 588 if needed.)*

2. After generating the simulated data, train and test SST:
   ```bash
   python deep_structural_embedding_prepost_shock.py
   ```

## Package List

re: 2.2.1
logging: 0.5.1.2
ipaddress: 1.0
zlib: 1.0
packaging: 23.1
numpy.version: 1.26.4
_ctypes: 1.1.0
ctypes: 1.1.0
mkl: 2.4.0
numpy.core._multiarray_umath: 3.1
numpy.core: 1.26.4
numpy.linalg._umath_linalg: 0.1.5
platform: 1.0.8
numpy: 1.26.4
PIL._version: 10.2.0
PIL: 10.2.0
defusedxml: 0.7.1
cffi: 1.16.0
PIL.Image: 10.2.0
PIL._deprecate: 10.2.0
numpy._core._multiarray_umath: 3.1
pyparsing: 3.1.4
cycler: 0.12.1
dateutil: 2.8.2
kiwisolver._cext: 1.4.7
kiwisolver: 1.4.7
matplotlib: 3.9.2
_shaded_ply: 3.7
_shaded_ply.lex: 3.8
_shaded_ply.yacc: 3.8
urllib.request: 3.11
_shaded_thriftpy: 0.4.13
json: 2.0.9
_decimal: 1.70
decimal: 1.70
xmlrpc.client: 3.11
socketserver: 0.4
http.server: 0.6
pkg_resources._vendor.more_itertools: 9.1.0
pkg_resources.extern.more_itertools: 9.1.0
pkg_resources._vendor.platformdirs.version: 2.6.2
pkg_resources._vendor.platformdirs: 2.6.2
pkg_resources.extern.platformdirs: 2.6.2
pkg_resources._vendor.packaging: 23.1
pkg_resources.extern.packaging: 23.1
IPython.core.release: 8.20.0
traitlets._version: 5.7.1
traitlets: 5.7.1
argparse: 1.1
executing.version: 0.8.3
executing: 0.8.3
six: 1.16.0
pure_eval.version: 0.2.2
pure_eval: 0.2.2
stack_data.version: 0.2.0
stack_data: 0.2.0
pygments: 2.15.1
decorator: 5.1.1
wcwidth: 0.2.5
prompt_toolkit: 3.0.43
parso: 0.8.3
jedi: 0.18.1
IPython: 8.20.0
colorama: 0.4.6
torch.version: 2.1.0
torch.torch_version: 2.1.0
tqdm._dist_ver: 4.65.0
tqdm.version: 4.65.0
tqdm.cli: 4.65.0
tqdm: 4.65.0
mpmath: 1.3.0
sympy.release: 1.12
sympy.multipledispatch: 0.4.9
sympy: 1.12
torch: 2.1.0
pytz: 2023.3.post1
numexpr: 2.8.7
bottleneck: 1.3.7
_csv: 1.0
csv: 1.0
pandas._version_meson: 2.1.4
pandas: 2.1.4
attr: 23.1.0
attrs: 23.1.0
idna.package_data: 3.4
idna.idnadata: 15.0.0
idna: 3.4
rfc3986_validator: 0.1.1
rfc3339_validator: 0.1.4
jsonschema: 4.19.2
markupsafe: 2.1.3
jinja2: 3.1.3
toolz: 0.12.0
altair: 5.0.1
fsspec: 2024.2.0
_brotli: 1.0.9
brotli: 1.0.9
urllib3._version: 2.1.0
urllib3.util.ssl_match_hostname: 3.5.0.1
urllib3.connection: 2.1.0
urllib3: 2.1.0
charset_normalizer.version: 2.0.4
charset_normalizer: 2.0.4
requests.packages.urllib3._version: 2.1.0
requests.packages.urllib3.util.ssl_match_hostname: 3.5.0.1
requests.packages.urllib3.connection: 2.1.0
requests.packages.urllib3: 2.1.0
requests.packages.idna.package_data: 3.4
requests.packages.idna.idnadata: 15.0.0
requests.packages.idna: 3.4
certifi: 2024.08.30
requests.__version__: 2.31.0
requests.utils: 2.31.0
socks: 1.7.1
requests: 2.31.0
portalocker.__about__: 2.8.2
portalocker: 2.8.2
torchdata.version: 0.7.0
torchdata: 0.7.0
filelock.version: 3.13.1
filelock: 3.13.1
torchtext.version: 0.16.0+cpu
torchtext: 0.16.0+cpu
thinc.about: 8.2.2
srsly.ujson.ujson: 1.35
srsly.cloudpickle: 2.2.0
srsly.ruamel_yaml: 0.16.7
srsly.about: 2.4.8
srsly: 2.4.8
pydantic: 1.10.12
thinc: 8.2.2
wasabi.about: 0.9.1
wasabi: 0.9.1
cymem.about: 2.0.6
preshed.about: 3.0.6
murmurhash.about: 1.0.7
spacy.about: 3.7.2
click: 8.1.7
shellingham: 1.5.0
mdurl: 0.1.0
markdown_it: 2.2.0
typer: 0.9.0
spacy: 3.7.2
setuptools._distutils: 3.11.7
setuptools.version: 68.2.2
setuptools._vendor.packaging: 23.1
setuptools.extern.packaging: 23.1
setuptools._vendor.more_itertools: 8.8.0
setuptools.extern.more_itertools: 8.8.0
setuptools._vendor.ordered_set: 3.1
setuptools.extern.ordered_set: 3.1
setuptools: 68.2.2
distutils: 3.11.7
GPUtil.GPUtil: 1.4.0
GPUtil: 1.4.0

