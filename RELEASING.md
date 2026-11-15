# Releasing polysmelt

Versions and releases come from commit messages. After CI passes on a push to `main`, the
[Auto release workflow](https://github.com/Azazel-Labs/polysmelt/actions/workflows/auto-release.yml)
reads the messages:

| Tag in a commit message | Effect |
| --- | --- |
| `#patch`, `#minor`, `#major` | Bumps the version in Cargo.toml and the lockfiles and commits "Bump version to X" to `main`. Counts commits since the last version change; the largest bump wins. |
| `#release` | Tags the current version (`vX.Y.Z`) and publishes it to crates.io. Counts commits since the last version tag. |

Put tags on a line of their own, alone or together, usually the last line of the message. A tag
inside a sentence is ignored, so describing these tags in a commit message is safe.

Both can go in one push: a `#minor #release` line bumps 0.1.0 to 0.2.0 and publishes it. Bump
without `#release` to let a version collect changes before releasing it. A `#release` with no bump
for a version that is already tagged only logs a warning.

After a bump, pull before your next push (`git pull --rebase`), since the bot's commit is on
`main`. If CI fails, nothing happens; the tags still count once a later push passes CI.

## crates.io

There is one crate, `polysmelt`. The
[Publish to crates.io workflow](https://github.com/Azazel-Labs/polysmelt/actions/workflows/crates.yml)
publishes it with the `CARGO_REGISTRY_TOKEN` repository secret. It checks that the tag matches
Cargo.toml's version and skips a version that is already published, so it can be rerun. To retry,
use **Actions → Publish to crates.io → Run workflow** with an existing tag such as `v0.1.0`.
Published versions can't be overwritten: a fix needs a new version.
