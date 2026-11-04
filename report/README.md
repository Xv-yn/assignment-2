# Supplied report sources

`original/latex_template.tex` and `original/latex_template.sty` were extracted
byte-for-byte from `LaTeX-template.zip`. `latex_template.sty` is an unchanged
working copy. Build metadata and macOS archive artifacts were omitted.

The full report is a later task. `preprocessing.tex` contains the Part 5 setup
table for inclusion then. See `../docs/design.md` for method references and
`../docs/data_audit.md` for verified input details.

Before compiling the actual report, override the style's 10 pt normal text and
17 pt title to the brief's 12 pt body and 14 pt title. Merely using the article
class's 12 pt option does not fix the style's own definitions. An exact Times
New Roman font is not installed locally; no compliant PDF is claimed. Preserve
the original sources and document any working overrides separately. Conservatively
count the execution appendix within the ten pages, excluding only references.
