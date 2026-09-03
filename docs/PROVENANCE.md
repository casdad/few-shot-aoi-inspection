# Public asset and data provenance

This repository publishes source code and configuration templates, not datasets or
trained AOI model bundles.

## Third-party code

The local DINOv2 inference subset under `assets/dinov2/` is pinned to upstream commit
`7764ea0f912e53c92e82eb78a2a1631e92725fc8` and retains its Apache-2.0 license and
model card in that directory.

## Expected public pretrained files

The training configuration expects these files, which are not committed:

| File | Expected SHA-256 | Source |
| --- | --- | --- |
| `assets/dinov2_vits14_pretrain.pth` | `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9` | Meta AI DINOv2 |
| `assets/resnet18-f37072fd.pth` | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` | TorchVision ResNet18 |

`assets/source_model.pt` and every adapted task bundle are also excluded. The local
source model available during release preparation did not contain a complete,
independently verified training-data provenance receipt, so it is not suitable for a
public release.

## Datasets

The release was exercised locally with DAGM2007, a synthetic industrial inspection
benchmark published by the Heidelberg Collaboratory for Image Processing. Its Zenodo
record states that the data is publicly available, but the record's explicit license
field was blank when checked for this release. Consequently, no DAGM image or derived
demo pixel is redistributed here. MVTec AD and all private/product datasets are also
excluded.

Users are responsible for downloading data from the original publisher and complying
with the current license and usage terms.
