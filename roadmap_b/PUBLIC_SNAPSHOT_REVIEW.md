# Public snapshot review — 2026-09-12

This is the first substantive source/evidence commit on top of an existing README-only Git commit, not a final submission or release. The current implementation is under `roadmap_b/`; older duplicate workspace trees and the compute/continuation filesystem are not published.

## Validation performed for publication

- A fresh directory was populated from the Git index, with no model/data artifact directories copied into it.
- Ordinary tests passed there: **58 passed, 3 skipped**, with five existing attention-mask deprecation warnings. The three opt-in real-pretrained tests were not rerun for documentation/export preparation. The synthetic 32-example integration test is a software check, not a new MELD experiment.
- Ruff lint and formatting checks passed (53 Python files).
- Historical preflight comparison: 64 of its 65 listed files stayed byte-identical; the only changed listed file was `recording_kit/README.md`. Frozen perception source, tests, configuration and recorder runtime were unchanged. New publication exporter/documentation are outside that old preflight identity.
- All 148 allowlisted original evidence file hashes and exported hashes were checked. Originals were not rewritten. Display-path redactions and the one privacy projection are enumerated in `evidence/PUBLIC_EXPORT_MANIFEST.json`.
- Staged text was scanned for obvious private-key headers, common credential/token forms and personal machine/SSH connection paths. Filename/size checks excluded private media, model/data blobs, environments and private notes. This is a concrete publication review, not a universal guarantee that automated scans detect every possible secret.
- The existing README-only ancestor was also reviewed before committing.

`.gitattributes` preserves exact historical source/evidence bytes rather than normalizing line endings behind integrity hashes. Generated SVG whitespace and CRLF line endings are accepted explicitly; model numerical tolerances are unchanged.

## Deliberately outside Git

Raw MELD media/archive extractions, processed audio, full annotations/transcripts, all checkpoints and cached representations, S01/S02 recordings, consent/timestamp metadata, private listener mappings/responses, provisional raw `outputs.json` and `recordings.jsonl`, private learning notes, local logs/audits, legacy duplicate source trees, environment caches and continuation archives.

Public model/policy hashes do not imply that task weights are distributed. Exact inference still requires the privately preserved operational A/B assets described in the implementation README. Historical failed numerical/evaluation attempts remain in the local continuation; small explanation/repair artifacts are exported where useful.

## Unchanged scientific status

**PERCEPTION ARCHITECTURE SEARCH COMPLETE.** Selected B seed 1337 is unchanged. Development evidence is not official-test evidence. The provisional delivery diagnostic found B less sensitive than fixed A; no tuning or reselection followed. L01 remains pending. No final freeze, official-test inference, model training, remote push or release tag was performed during this publication task.

The later current-source preflight must incorporate new tooling/documentation before final freeze. This publication check does not substitute for that guarded evaluation workflow.
