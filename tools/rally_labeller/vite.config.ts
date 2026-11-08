import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

import { labellerApi } from "./server/labellerApi.ts";

const repoRoot = path.resolve(fileURLToPath(new URL(".", import.meta.url)), "../..");

// Relative folders count from the repository root, like the Python scripts run from there.
const folder = (variable: string, fallback: string) => path.resolve(repoRoot, process.env[variable] ?? fallback);

export default defineConfig({
  plugins: [
    labellerApi({
      mediaRoot: folder("RALLY_LABELLER_MEDIA", "local_scratch/rally_labeller"),
      labelsRoot: folder("RALLY_LABELLER_LABELS", "data/amateur_contacts"),
      sourcesRoot: folder("RALLY_LABELLER_SOURCES", "local_scratch/amateur_vids"),
      repoRoot,
    }),
  ],
});
