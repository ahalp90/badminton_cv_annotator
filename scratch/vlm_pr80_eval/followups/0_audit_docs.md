# Step 0: audit the completed investigation docs

## Goal

Check the attached documentation against the full retained experiment history before running anything new.

Do **not** run new experiments yet.

Use the repository's prompts, manifests, outputs, human truth and scoring code as the source of truth.

## Check

For each material claim, ask:

- Was this actually tested?
- Is the result attributed to the correct model?
- Are the counts, denominators and ground-truth mappings correct?
- Is a measured fact being turned into a broader claim than the experiment supports?
- Is an important caveat missing?
- Is anything technically true but likely to give the reader the wrong impression?

Pay particular attention to Intern-only versus Qwen-only results.

Keep the completed investigation separate from proposed future work.

## Output

Return only:

1. factual corrections needed;
2. materially missing context;
3. wording that is technically true but misleading;
4. otherwise, a clear statement that the docs are faithful.

Preserve the current simple writing style. Follow [`../WRITEUP_PRINCIPLES.md`](../WRITEUP_PRINCIPLES.md). Do not add detail merely because it exists.

After corrections are accepted, freeze the completed-investigation docs before starting Follow-up 1.
