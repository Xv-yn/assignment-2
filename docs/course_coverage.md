# Course coverage review — 28 October 2026

**Decision:** retain forward correction as the course-taught method and use
**generalized cross-entropy (GCE), gamma=0.7**, as the method outside the supplied
course content. This conclusion covers the provided Week 2–11 lecture PDFs and
accompanying HTML notes, rather than unprovided tutorials, recordings or weeks.

## Positive evidence of taught methods

| Materials | Content actually taught | Consequence |
| --- | --- | --- |
| [W6.pdf](../course-content/W6.pdf), PDF pages 33, 41, 45–48 | Least-squares, absolute, Cauchy and correntropy/Welsch losses; robustness of surrogate losses | Robust losses in general are taught; that alone does not establish coverage of GCE |
| [W8.pdf](../course-content/W8.pdf), PDF pages 29 and 32 | Symmetric 0–1, unhinged, sigmoid and ramp losses; importance reweighting, unbiased estimators, cost-sensitive losses and rank pruning | Do not present these as the outside-course method |
| [W8.pdf](../course-content/W8.pdf), PDF pages 40–42 | Transition matrices, forward and backward learning | Forward correction is explicitly taught |
| [W9.pdf](../course-content/W9.pdf), PDF pages 11–14 and 15–19 | Forward/backward review, anchor-point assumption and transition-matrix estimation | The proposed anchor estimator is also taught; it is not the qualifying second classifier |

## Evidence for selecting GCE

All ten PDFs were converted with `pdftotext -layout`. Searches across every
extracted lecture and the accompanying HTML notes/lecture JavaScript found no
mention of generalized/generalised cross-entropy, GCE, Sabuncu, Box–Cox, or the
Zhang–Sabuncu method. Unrelated Zhang references do occur and were distinguished
from this paper. The relevant robust-loss and label-noise sections were also
reviewed for the objective, rather than relying solely on its acronym.

The proposed loss is the untruncated family from
[Zhang and Sabuncu, NeurIPS 2018, Section 3.2](https://arxiv.org/html/1805.07836):

`L_gamma(p,S) = (1 - p[S]**gamma) / gamma`, with `0 < gamma <= 1`.

The pilot uses gamma=0.7. This specific interpolation between categorical CE
and MAE was not found in the supplied materials. The lecture's discussion of
robust/symmetric losses is related background, rather than coverage of this
particular objective. We use standard GCE, not the paper's truncated variant.
Both methods share a CNN but have distinct objectives and noise treatments.

Suggested report wording: “Forward correction follows the course's treatment
of transition-matrix learning. Our second method is generalized cross-entropy
(Zhang and Sabuncu, 2018), which is not covered in the supplied Week 2–11 lecture
materials.” If additional teaching materials become available, extend this
review; this record is not teaching-staff confirmation.

## Reviewed lecture inventory

Hashes identify the exact supplied files reviewed. Accompanying notes W2–W11
and their lecture JavaScript were searched as secondary material; the lecture
PDFs are the primary evidence. The supplied folder contains no Week 1 PDF or
separate tutorials.

| File | PDF pages | SHA-256 |
| --- | ---: | --- |
| [W2.pdf](../course-content/W2.pdf) | 66 | `fc3d90a20f7c677106a359ab9a9431cd62a28bfeaaf7c8e4fa178f2b7d7685a2` |
| [W3.pdf](../course-content/W3.pdf) | 60 | `794a2cbe4dd564ceef78826602c9e87869b53daffc6823d6949d3daf74ddd8c7` |
| [W4.pdf](../course-content/W4.pdf) | 52 | `a18391ec4d3492b421537280e48fe352502d134833d5779922cc32f9109be112` |
| [W5.pdf](../course-content/W5.pdf) | 45 | `ad90f4af1241506a172504c5cc7739d5c9c1843c772fc0bc72371cb269d3ee9a` |
| [W6.pdf](../course-content/W6.pdf) | 54 | `b579568dc33a3a3a01e3857afc5f962265336040f990a4dd75484208b0afd369` |
| [W7.pdf](../course-content/W7.pdf) | 46 | `2cf608a552ceb7c7ff1bba9968d81866aef840bea4a982f043d49cfef398c2fa` |
| [W8.pdf](../course-content/W8.pdf) | 46 | `af8c963f3ea5268b9a914712c632048bd0afa24f493b36a54ff721e0016c7739` |
| [W9.pdf](../course-content/W9.pdf) | 55 | `fac8bbd67fb84f11c2f277130bc1207730779a052f810e1cfa2bd30d58652fa5` |
| [W10.pdf](../course-content/W10.pdf) | 50 | `fa0858f2f6b8b882dcd5e9a815d688ea58f7ee0757ec0ce80a391bf9b0c31845` |
| [W11.pdf](../course-content/W11.pdf) | 41 | `3234eb710b3d09fd12b8a42a7465e526a08d93eb5403ba1e01128dfd203822ff` |
