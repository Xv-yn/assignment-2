# Supplied-data audit

Completed 22 October 2026 using the isolated environment recorded in
[data_audit.json](data_audit.json). Full per-array ranges, shapes, dtypes,
finite checks, class frequencies, preprocessing, grid indices, and dataset
SHA-256 hashes are recorded there. No classifier was trained or tested.

| Dataset | Xtr | Str | Xts | Yts | Noisy class counts (0, 1, 2) |
| --- | --- | --- | --- | --- | --- |
| FashionMNIST0.3 | (18000, 784), uint8 | (18000,), uint8 | (3000, 784), uint8 | (3000,), uint8 | 6092, 5930, 5978 |
| FashionMNIST0.6 | (18000, 784), uint8 | (18000,), uint8 | (3000, 784), uint8 | (3000,), uint8 | 5918, 5970, 6112 |
| CIFAR | (15000, 32, 32, 3), uint8 | (15000,), int64 | (3000, 32, 32, 3), uint8 | (3000,), int64 | 5000, 5000, 5000 |

All archives contain exactly Xtr, Str, Xts, and Yts. All arrays have finite
numeric values; features lie in [0,255] and labels contain exactly IDs 0, 1, 2.
Feature/label counts match in each partition. Stored dimensions and dtypes
match the assignment's verified dataset contract.

The checked batches have shapes (32,1,28,28) for FashionMNIST and (32,3,32,32)
for CIFAR, contiguous float32 images in [0,1], and int64 labels. Repeated calls
to the pool and test transforms are identical. Clean test arrays were opened
only for integrity and deterministic-transform checks; no model selection,
class semantics, normalization fit, or performance calculation used them.

## Visual inspection and transform decision

Inspected these generated grids, each containing four examples per observed
noisy ID, sampled with seed 2026 from Xtr only:

- [FashionMNIST0.3](../results/smoke/FashionMNIST0.3/training_grid.png)
- [FashionMNIST0.6](../results/smoke/FashionMNIST0.6/training_grid.png)
- [CIFAR](../results/smoke/CIFAR/training_grid.png)

FashionMNIST has coherent grayscale shape detail, and CIFAR has coherent RGB
images with no apparent channel-order distortion. The examples within a noisy
ID visibly vary, so these grids cannot establish a clean semantic name for
that ID. Captions therefore use observed IDs only. These are samples of the
combined noisy pool, not an already-created 80% training split.

Use deterministic division by 255 for all methods and partitions. Preserve
native resolution. Do not crop, flip, rotate, or aggressively distort the
initial FashionMNIST inputs, whose small shape details can be lost. Crop/flip
augmentation for CIFAR is deferred: this small grid does not establish
label preservation for the clean classes. No augmentation is used initially.
No channel standardization is used, so there are no fitted mean/std values
and no risk of estimating those statistics from the future validation pool.
Every run records the fixed preprocessing specification with fitted statistics
explicitly null. If standardization is added later, its statistics must be
fit only on the current 80% training partition and saved with that run.

The report-ready operation table is [preprocessing.tex](../report/preprocessing.tex).
The supplied-data archive is unchanged; labels were neither corrected nor
replaced, and no external data was acquired.
