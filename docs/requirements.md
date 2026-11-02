# Requirements and current acceptance status

Reference: `instructions.md`, supplied `assignment2_26.pdf`, and `todo.md`.
Updated 28 October 2026. This is an acceptance map, not a claim that all deliverables are complete.

Implemented: saved ten-seed 80/20 splits, the shared CNN, GCE and supplied-T forward pilots, noisy-only validation and best checkpoints. GCE is outside the reviewed W2–W11 content; see [course_coverage.md](course_coverage.md). All 24 tests pass. CIFAR estimation and final clean-test experiments remain; see [training_protocol.md](training_protocol.md).

| Deliverable | Acceptance condition | Scheduled implementation |
| --- | --- | --- |
| Robust classification | Two distinct noise-robust objectives on all three datasets; at least one method verified outside course coverage | Parts 7–8, 11 |
| CIFAR transition estimation | Estimate separately using each training partition; feed that exact matrix to forward correction | Part 9 |
| Repeated protocol | At least ten different random 80/20 holdouts per method/dataset; noisy validation, clean final test | Parts 6, 10–11 |
| Main results | Six combinations, each with mean and SD of clean-test top-1 accuracy and retained per-run evidence | Part 11 |
| Report | Supplied LaTeX template; abstract, introduction, related work, methods, experiments, conclusion, references, execution appendix | Parts 14–15 |
| Formatting | 14 pt title, 12 pt body, Times New Roman; ten pages excluding references, conservatively including appendix | Part 15 |
| Submission | PDF and source ZIP; `algorithm/` plus an empty `data/`, without datasets | Parts 16–17 |

Prioritize the 45 marks for known-noise classifiers (23), estimation (12), and
unknown-noise classification (10). Preserve experiment records and regenerate
tables from them. Ten runs is a minimum, not evidence by itself that a method
is correct. The remaining report marks assess prior work, clear presentation,
depth, introduction, abstract, and conclusions; code contributes another 20.

No final models, transition estimates, accuracy results, report PDF, or submission
ZIP are claimed here. Group formation, identifying details, allocation, and peer
review were skipped at the user's request.
