# Releasing polysmelt

Versions and releases come from commit messages. After CI passes on a push to `main`, the
[Auto release workflow](https://github.com/Azazel-Labs/polysmelt/actions/workflows/auto-release.yml)
reads the messages:

| Tag in a commit message | Effect |
| --- | --- |
| `#release` | Tags the next version (`vX.Y.Z`) and publishes it to crates.io. Counts commits since the last version tag. |
| `#patch`, `#minor`, `#major` | How far that release moves the version from the last tag; the largest wins, and patch when none says. On its own it does nothing: it waits for a `#release`. |

Put tags on a line of their own, alone or together, usually the last line of the message. A tag
inside a sentence is ignored, so describing these tags in a commit message is safe.

A `#minor #release` line releases 0.1.0 as 0.2.0. Leave out `#release` to let changes collect;
the bump lines count once a later push releases.

Nothing is committed to `main`, so there's never a bot commit to pull. The version is set in
Cargo.toml and the lockfiles in a commit of its own that only the version tag points to, which is
what's published. `main`'s Cargo.toml keeps the version it was last given by hand; the version tags
are what count. If CI fails, nothing happens; the tags still count once a later push passes CI.

## crates.io

There is one crate, `polysmelt`. The
[Publish to crates.io workflow](https://github.com/Azazel-Labs/polysmelt/actions/workflows/crates.yml)
publishes it with the `CARGO_REGISTRY_TOKEN` repository secret. It checks that the tag matches
Cargo.toml's version and skips a version that is already published, so it can be rerun. To retry,
use **Actions → Publish to crates.io → Run workflow** with an existing tag such as `v0.1.0`.
Published versions can't be overwritten: a fix needs a new version.
