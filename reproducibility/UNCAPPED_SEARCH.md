# Uncapped search in the earlier executable code

The exp040 worker now evaluates every ranked prefix, k=1..N, of its complete
adjacent-layer candidate pool. The historical four-prefix limit is removed.
For two layers with eight heads N=8; for four layers N=24. The ranking,
50% discovery-coverage target, shortest-prefix rule, fallback, protection loss,
training settings and confirmation split retain their definitions. Confirmation
does not reselect paths. The exp047 utility loop also evaluates all its ranked
prefixes; its experimental non-adjacent operator has no manuscript validation
claim. The reported four-layer experiments use adjacent pairs 1->2, 2->3, 3->4.

## Historical selection evidence

`uncapped_selection_validation.json` records the ten exp040 two-layer selections
and their four saved discovery-prefix coverages, extracted from the retained
analysis summary identified by SHA-256. Nine runs first reach 50% at k=1.
2L/128 s17 first reaches it at k=2 (46.8223%, then 82.9414%), selecting IDs [1,6].
Later prefixes cannot change an already observed shortest prefix reaching the
target. Thus those ten selections are invariant to removing the old limit,
provided the original inputs, fork, singleton ranking and coverage computation
are kept fixed.

This check reads saved discovery summaries. Raw discovery arrays and original
path-selection receipts were not found in the retained workspace, so their
ranking, raw-array hashes and coverage values have not been independently
recalculated. The source change does not represent a rerun of discovery or
protection training, and it does not rewrite the original execution history.

The earlier primary s5 protocol fixes Head 0. The independent s7 protocol selects
one head by singleton discovery score (reported Head 1). The fixed-start
s18--s22 protocols also rank singleton heads. Their discovery records needed to
establish equivalence with the 50% shortest-prefix rule are absent from the
retained workspace. These seven trajectories remain **unverified** for that
equivalence; their historical implementations are retained. Their final accuracy
and held-out confirmation coverage cannot settle discovery selection.

No new training was run. The existing exp040 results can be retained on the
saved-summary selection argument above. A claim of one retrospectively verified
selection rule across all earlier experiments still requires the missing
discovery evidence. If that evidence selects different paths, the affected
protection runs and dependent results must be regenerated.

## Source checks and limits

`python3 tools/verify_package.py` executes the actual exp040 discovery function
with synthetic records and no scientific imports. Cases include first reaching
the target at k=5, reaching it at k=1 while still evaluating all 24 prefixes,
no prefix reaching the target, and all coverages undefined. The check also
validates the ten evidence rows against the released pathway IDs.

The existing PR control still requires an equally sized disjoint set with the
same per-source-layer counts. If too few non-selected heads remain, its existing
explicit error applies. Removing the search cap does not change that control's
definition or truncate the selected prefix to fit it. None of the ten retained
two-layer selections encounters this condition.

Updated source/configuration manifests identify the changed executable release.
The original bytes remain at Git revision
`2b30428`, and the change record retains the previous hashes of changed files.
Use a new run root with the updated code identity; historical checkpoint and
receipt hashes continue to identify the original execution.
