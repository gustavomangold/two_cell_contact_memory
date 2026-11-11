This is the published engine for DOI 10.1038/s42005-026-02619-2.

`upstream/` is an unchanged copy of `src/` from
https://github.com/mb176/cell_simulation at commit
`fa2989f7841ea4c8aea0d5691ed0710ee9432da4`. File hashes are in `provenance.json`.

`driver.c` calls the original initialization and `SingleStep`, adding compact
contact measurements and full-precision initial/final state output. The build
copy replaces the wall-clock seed with `PAPER_SEED`; tracked upstream files stay
unchanged. No force, state-update, RNG or GSL minimization functions are replaced.
A build copy renames the original neighbor-list entry point to allow dispatch:
boxes with fewer than three bins per side use unique all-pairs discovery; larger
boxes retain the original list and order. Tracked upstream files stay unchanged.
No fast-math is enabled. See `docs/paper.md` at the repository root.
