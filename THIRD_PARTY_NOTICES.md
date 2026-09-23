# Third-party notices

## DeepCASE

This project interoperates with the public DeepCASE implementation:

- Repository: <https://github.com/Thijsvanede/DeepCASE>
- Paper: *DeepCASE: Semi-Supervised Contextual Analysis of Security Events*
- Upstream license: MIT License
- Pinned revision in `requirements.txt`: `7d19eaa798da257b83567aeaed78d88bd3574370`

The upstream DeepCASE repository is installed as a dependency. Its source tree is not copied into this repository.

## AIT-ADS

Some experiment scripts expect the public AIT Alert Dataset (AIT-ADS):

- Dataset record: <https://doi.org/10.5281/zenodo.8263181>

The dataset is not redistributed here. Users must obtain it from the official record and follow the terms stated there.

## HDFS benchmark data

The long-context experiment contains adapters for the public HDFS benchmark format used by DeepCASE. No HDFS data is redistributed in this repository. Refer to the upstream DeepCASE repository for its data preparation instructions and original sources.
