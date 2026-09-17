# Crossed audio retention

This pilot tests whether audio--text agreement can allocate a fixed acoustic
token budget better than uniform retention or an audio-only policy.

**Result:** agreement-adaptive retention improved the point estimate by 7.96%
over fixed retention but was slightly worse than the audio-only control, with
uncertainty spanning zero. See [RESULTS.md](RESULTS.md) for the full summary.

The A100 stage crosses three transcript conditions with clean and degraded
audio for twelve RAVDESS recordings. It preserves every transcript embedding
and evaluates zero, two, or four appended chronological mean-pooled acoustic
positions at fusion strength one. Actors 01--04 train the policies; actors
05--06 remain held out.

One VIPER run connects the A100 measurement to a CPU verification stage. The
verification stage hashes the exact input artifact and recomputes both fitted
models, all allocations, equal-budget checks, clustered contrasts, cell
summaries, and bootstrap values. The retained run directory is a Git audit
copy; native replay still requires the originating VIPER local store or VIPER
Cloud.

The result is exploratory. Each allocation unit is one recording, transcript
condition, and audio condition; this pilot does not allocate positions among
several segments inside one prompt.
