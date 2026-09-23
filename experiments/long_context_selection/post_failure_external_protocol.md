# Post-failure safety refinement and external HDFS protocol

This protocol is a new, explicitly post-failure track. It was written after
the frozen AIT-ADS `association_abs10` confirmation failed its false-escalation
criterion, but before computing any `anchor5_assoc5` outcome or downloading
the HDFS files. It cannot alter or rescue that failed confirmation.

## Mechanistic change

The failed selector could replace every recent event with an older event that
had a large training association. Matched total review workload did not prevent
those replacements from turning benign alerts into automatic Incident
decisions. The fixed refinement retains the newest five source slots
unconditionally and fills exactly five remaining slots from the older 95
candidates using absolute training-only association evidence. Ties prefer the
more recent candidate and the ten selected positions are returned in temporal
order. No target or predecessor outcome is an inference-time feature.

The 5/5 allocation is fixed by symmetry and is not tuned over the six already
observed AIT confirmation scenarios. Those six scenarios may be reported only
as post-hoc mechanism checks. The only AIT development scenarios used for a
go/no-go decision are the original `fox` and `russellmitchell` scenarios. The
method advances to HDFS only when each has positive three-seed mean strict
recall gain and their pooled false-escalation relative increase is at most 5%.

## New external data

HDFS is the only public reproducibility dataset named by the original DeepCASE
authors; their Lastline dataset is unavailable under NDA. We use the three
preprocessed files linked by the authors from the DeepLog repository. One line
is one HDFS block sequence. Exact duplicate lines must not cross partitions.
The SHA-256 hash of the exact line modulo ten assigns official test sequences:
buckets 0--3 development, 4--5 validation, and 6--9 final test. File identity
provides only the development/validation/test outcome (`normal=0`,
`abnormal=1`); final-test outcomes are not used to fit the association table,
event vocabulary, context model, cluster policy, or operating point.

The official `hdfs_train` file is normal training data. Candidates are the 100
preceding events in the same line/sequence; HDFS has no timestamps, so this
external experiment does not validate the AIT one-day timeout. Both methods
still supply exactly ten positions to the same DeepCASE architecture.

## Evaluation and stopping rule

The baseline uses the original `(eps=.1, threshold=.2)` point. A selector point
is eligible only if its validation false-escalation rate is no more than 1.05
times the baseline validation rate. Among eligible points, final labels remain
hidden and only closeness to the baseline final-stream review workload is used;
ties prefer lower workload and then frozen grid order. No eligible point means
failure for that seed.

Primary recall credits only automatic Anomaly decisions; Reject is unresolved
and adds to review workload. Success requires a positive recall-gain 95%
paired bootstrap interval resampled by final HDFS sequence, no more than 5%
pooled relative false-escalation increase, exactly ten context positions, and
all final sequences. Three neural seeds are technical repeats and are averaged
within sequence before bootstrapping; they are not treated as independent
datasets. A failure is reported as failure, not repaired by a different
post-hoc threshold or split.
