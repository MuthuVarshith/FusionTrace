# Purging old model weights from Git LFS history

**Status: prepared, NOT executed.** This rewrites history for everyone. Run it
only after the v2 branches are merged, and announce a freeze first.

Since v2, `backend/models/` is untracked and gitignored. Weights come from the
Hugging Face Hub (`backend/scripts/download_models.py`). Older commits still
reference the LFS objects (`backend/models/audio_model/model.safetensors` and
`backend/models/image_model/EfficientnetV2_model.pth`; the pre-upgrade
assessment measured the weights at about 814 MB), so every clone still downloads them.

## 0. Preconditions

- Every weight bundle the detectors need is on the Hub, and
  `python scripts/download_models.py --all` works on a fresh clone. If the
  legacy weights are not on the Hub yet, upload them first, for example
  `hf upload your-user/FusionTrace-models backend/models/image_model image_model`.
- All open branches and PRs are merged or noted (their SHAs will change).
- You have admin rights on the GitHub repository (to bypass branch protection).

## 1. Back up

```bash
git clone --mirror https://github.com/MuthuVarshith/FusionTrace.git FusionTrace-backup.git
cd FusionTrace-backup.git
git lfs fetch --all          # the LFS objects themselves, not just pointers
cd ..
tar czf FusionTrace-backup.tgz FusionTrace-backup.git
```

Keep the archive off the machine until the rewrite is verified.

## 2. Rewrite (git filter-repo, recommended)

```bash
pip install git-filter-repo
git clone --mirror https://github.com/MuthuVarshith/FusionTrace.git FusionTrace-rewrite.git
cd FusionTrace-rewrite.git
git filter-repo --path backend/models --invert-paths
```

This removes `backend/models/` from every commit on every branch and tag.
filter-repo drops the `origin` remote afterwards; re-add it in step 4.

**Alternative: BFG Repo-Cleaner** (needs Java):

```bash
java -jar bfg.jar --delete-files "{model.safetensors,EfficientnetV2_model.pth}" FusionTrace-rewrite.git
cd FusionTrace-rewrite.git
git reflog expire --expire=now --all && git gc --prune=now --aggressive
```

BFG matches file names anywhere in the tree and leaves HEAD alone (HEAD no
longer contains them). It does not remove the small JSON configs; filter-repo
does.

## 3. Verify

```bash
git log --all --oneline -- backend/models     # expect no output
git lfs ls-files --all                         # expect no output
git count-objects -vH                          # size-pack should be small
```

Clone the rewritten mirror into a scratch directory and run the test suite and
`download_models.py` once.

## 4. Force-push

```bash
git remote add origin https://github.com/MuthuVarshith/FusionTrace.git
git push --force --all origin
git push --force --tags origin
```

(Avoid `git push --mirror`: it also tries to push GitHub's read-only `refs/pull/*`.)

Implications:

- Every commit SHA changes. Links to commits in issues, PRs and docs break.
- Open PRs can no longer be merged cleanly; recreate them from rebased branches.
- Temporarily allow force pushes on protected branches, then restore protection.
- Forks keep the old history and the weights; their owners must re-fork.
- Signed commits lose their signatures.

## 5. Collaborators

Everyone must **re-clone**. Do not `pull`, `merge` or push from an old clone:
that reintroduces the purged blobs. Rebase personal branches onto the new
history with `git rebase --onto`, or re-apply patches (`git format-patch`).
Local worktrees made from the old clone should be removed too.

## 6. Reclaim GitHub LFS storage

Rewriting history does not delete objects from GitHub's LFS store; they keep
counting against storage quota. At the time of writing, GitHub's documentation
("Removing files from Git Large File Storage") says the way to remove LFS
objects is to delete and recreate the repository, which also deletes issues,
stars, PRs and wiki. Check the current docs first. If you go that way:

1. Export issues/PRs you want to keep.
2. Delete the repository, recreate it with the same name.
3. Push the rewritten mirror (step 4).

## 7. Local cleanup

In any clone you keep:

```bash
git lfs prune
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```
