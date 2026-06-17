# Certus DCASE Gunshot Classification Audio ML Model Training Code

Contains all code and configurations necessary to replicate the results of Certus Innovations' 2026 DCASE submission.

## Abstract

Acoustic gunshot detection is a problem with applications across civilian public safety, military operations, and wildlife conservation, yet the field lacks a rigorous exploration of feature extraction techniques with a focus on generalization to realistic data. The mixed effectiveness of commercial gunshot detection and classification systems indicates an open problem that is not adequately addressed by the current literature. In this paper, we present a systematic investigation of common feature extraction techniques using a dataset of 23,000 gunshot recordings across 85 firearms and 21 calibers. We benchmark three feature extraction techniques with 12 total unique parameter sets using ResNet-18. Our results demonstrate that using the correct feature extraction technique can improve top-1 accuracy by up to 20%, and utilizing the correct parameters for a given feature extraction technique can improve that value by up to 4.7%.

## Configuration and Setup

Install the dependencies using `pip`:

```bash
pip install -r requirements.txt
```

These experiments were performed on a workstation with 64 GB of RAM, an Intel i9-12900K, and an NVIDIA GeForce GTX 4060.

## Usage

To replicate our results, you can either use the parameter files provided in the `params/` directory (after updating the `METADATA` value), or write your own as described in [`scripts/full_training.py`](scripts/full_training.py#L9).

```bash
python3 scripts/full_training.py params/exp1_stft_balanced.json
```

A single run of this script will execute the training code 5 times with different random seeds and output the results to the `logs/` folder. Expect each full replication run to take ~5 hours.

## Authors

- **Sinclair Gurny** - _Initial work_
- **Ryan Quinn** - _Refactoring and documentation_

## License

MIT.

## Citing

TODO.
