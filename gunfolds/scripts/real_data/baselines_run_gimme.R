#!/usr/bin/env Rscript
# =============================================================================
# baselines_run_gimme.R
# =============================================================================
# Pooled, group-level GIMME over a folder of per-subject CSVs, for the FBIRN
# baseline pipeline (driven by baselines_fmri_experiment.py).
#
#   Rscript baselines_run_gimme.R <in_dir> <out_dir> [ar] [groupcutoff] [subcutoff]
#
#   in_dir   : folder of per-subject CSVs (rows = time, cols = V1..VN), one file
#              per subject (written by `--method GIMME --stage export`).
#   out_dir  : gimme output dir.  The key artifact consumed downstream is
#              <out_dir>/indivPathEstimates.csv (long format:
#              file, lhs, op, rhs, beta/est, se, z, pval, ...), parsed by
#              `--method GIMME --stage collect`.
#   ar           : "TRUE"/"FALSE"  include the AR (lag-1 self) term (default TRUE)
#   groupcutoff  : proportion for a path to enter the group model (default 0.75)
#   subcutoff    : subgroup cutoff (default 0.50; subgroup is OFF here)
#
# GIMME is inherently a GROUP method: it estimates a shared group-level path
# structure, then frees subject-specific paths.  Running it once over the whole
# cohort (pooled) is the canonical usage and matches how we ran it before.
#
# Requires the `gimme` R package:  install.packages("gimme")
# =============================================================================

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("usage: Rscript baselines_run_gimme.R <in_dir> <out_dir> [ar] [groupcutoff] [subcutoff]")
}
in_dir      <- args[1]
out_dir     <- args[2]
ar          <- ifelse(length(args) >= 3, as.logical(args[3]), TRUE)
groupcutoff <- ifelse(length(args) >= 4, as.numeric(args[4]), 0.75)
subcutoff   <- ifelse(length(args) >= 5, as.numeric(args[5]), 0.50)

if (is.na(ar)) ar <- TRUE

suppressMessages(library(gimme))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

cat(sprintf("GIMME pooled run\n  in : %s\n  out: %s\n  ar=%s groupcutoff=%.2f\n",
            in_dir, out_dir, ar, groupcutoff))

fit <- gimme(
  output_dir  = out_dir,
  data        = in_dir,
  sep         = ",",
  header      = TRUE,
  ar          = ar,
  plot        = FALSE,
  subgroup    = FALSE,
  groupcutoff = groupcutoff,
  subcutoff   = subcutoff
)

# gimme writes indivPathEstimates.csv into out_dir automatically. Sanity-check.
est <- file.path(out_dir, "indivPathEstimates.csv")
if (file.exists(est)) {
  cat(sprintf("GIMME done. Wrote %s\n", est))
} else {
  cat("WARNING: indivPathEstimates.csv not found in out_dir; ",
      "check the gimme version's output file names.\n", sep = "")
  cat("Files present:\n")
  print(list.files(out_dir))
}
