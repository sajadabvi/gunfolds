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
# plot MUST default TRUE: gimme 0.7.x has a bug where the individual-level step
# (get.params) references `ind_plot_psi`, which is only created when plot=TRUE.
# With plot=FALSE it dies "object 'ind_plot_psi' not found". We don't use the
# PDFs it writes -- we only read indivPathEstimates.csv -- but plotting must be
# ON to avoid the crash. Pass "FALSE" as the 6th arg only on a gimme version
# that fixed this.
do_plot     <- ifelse(length(args) >= 6, as.logical(args[6]), TRUE)

if (is.na(ar)) ar <- TRUE
if (is.na(do_plot)) do_plot <- TRUE

suppressMessages(library(gimme))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

cat(sprintf("GIMME pooled run\n  in : %s\n  out: %s\n  ar=%s groupcutoff=%.2f plot=%s\n",
            in_dir, out_dir, ar, groupcutoff, do_plot))

fit <- gimme(
  out         = out_dir,        # gimme's output-dir arg is `out` (not output_dir)
  data        = in_dir,
  sep         = ",",
  header      = TRUE,
  ar          = ar,
  plot        = do_plot,        # TRUE to dodge the gimme 0.7.x ind_plot_psi bug
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
