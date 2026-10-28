# Final classification results

Clean test top-1 accuracy (%), sample SD (ddof=1).

| Dataset | Method | Runs | Mean (%) | SD (%) |
| --- | --- | ---: | ---: | ---: |
| CIFAR | forward | 10 | 70.513 | 6.684 |
| CIFAR | gce | 10 | 64.440 | 6.714 |
| FashionMNIST0.3 | forward | 10 | 99.063 | 0.136 |
| FashionMNIST0.3 | gce | 10 | 98.997 | 0.119 |
| FashionMNIST0.6 | forward | 10 | 93.087 | 1.548 |
| FashionMNIST0.6 | gce | 10 | 92.137 | 2.240 |

## CIFAR: estimated transition mean

```
[[0.43552 0.31118 0.2533 ]
 [0.24515 0.4429  0.31196]
 [0.35662 0.22966 0.41372]]
```

## FashionMNIST0.3: estimated transition mean

```
[[7.59590e-01 2.21842e-01 1.85674e-02]
 [7.54791e-04 7.27421e-01 2.71825e-01]
 [2.21375e-01 2.19111e-03 7.76434e-01]]
```
Mean entrywise absolute error: 0.04262; mean Frobenius error: 0.16778.

## FashionMNIST0.6: estimated transition mean

```
[[0.39636 0.30591 0.29773]
 [0.29934 0.39435 0.30631]
 [0.28554 0.27743 0.43703]]
```
Mean entrywise absolute error: 0.01789; mean Frobenius error: 0.06854.

Matrices are estimated independently per split. CIFAR classifiers use their own split's matrix, not this aggregate.
Check the course coverage requirement against your lecture material before submitting.
