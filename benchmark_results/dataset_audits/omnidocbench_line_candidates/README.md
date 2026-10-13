# OmniDocBench line-detection candidate audit

Candidate pages: 69. Review `previews/` and set `review` in `pages.csv` to `keep` or `drop`; `GALLERY.md` displays every page.

Selection requires at least one `text_span` and line spans for every top-level `text_block`.

- Green: `text_span`, proposed line ground truth.
- Red: page region without line spans, proposed ignore region.
- Orange: inline equation, footnote mark, or another non-text span.
