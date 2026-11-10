# M (2026-11-10): rank each exported observation matrix with the CRAN ScottKnottESD package (versions "p" and "np").
# Rscript log_parsing_skesd_r.R data/skesd_check   -> <name>.r.json per <name>.csv (columns = methods, rows = observations)
suppressMessages(library(ScottKnottESD, lib.loc = "~/R/lib"))
args <- commandArgs(trailingOnly = TRUE)
for (f in list.files(args[1], pattern = "\\.csv$", full.names = TRUE)) {
  df <- read.csv(f, check.names = FALSE)
  out <- c()
  for (v in c("p", "np")) {
    g <- sk_esd(df, version = v)$groups
    out <- c(out, sprintf('"%s": {%s}', v, paste(sprintf('"%s": %d', names(g), as.integer(g)), collapse = ", ")))
  }
  writeLines(paste0("{", paste(out, collapse = ", "), ", \"version\": \"", packageVersion("ScottKnottESD", lib.loc = "~/R/lib"), "\"}"),
             sub("\\.csv$", ".r.json", f))
}
