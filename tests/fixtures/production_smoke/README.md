# ClaimCI production smoke fixture

This private repository is a controlled, synthetic fixture for ClaimCI
production certification. It contains no customer data and executes no code.

The release-smoke pull request changes only the candidate result from a
baseline-equivalent score to a reproducible improvement. With the unchanged
configuration and evaluation dataset, the expected deterministic ClaimCI
verdict is `SUPPORTED`.
