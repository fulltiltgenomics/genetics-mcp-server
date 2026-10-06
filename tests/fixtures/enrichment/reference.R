suppressPackageStartupMessages(library(fgsea))
out <- "/out"
min_size <- 5
max_size <- 300

pathways <- gmtPathways(file.path(out, "gene_sets.gmt"))
universe <- readLines(file.path(out, "universe.txt"))
genes <- readLines(file.path(out, "gene_list.txt"))
ranked <- read.delim(file.path(out, "ranked_stats.tsv"))
gstats <- read.delim(file.path(out, "gene_stats.tsv"))

fmt <- function(df) {
  num <- vapply(df, is.double, TRUE)
  df[num] <- lapply(df[num], function(x) sprintf("%.17g", x))
  df
}
write_tsv <- function(df, name) {
  write.table(fmt(df), file.path(out, name), sep = "\t", quote = FALSE, row.names = FALSE)
}

# --- ORA ---------------------------------------------------------------------------------
# hypergeometric upper tail P(X >= k): sets and the gene list are restricted to `u`, and BH
# runs over the sets that pass the size filter after that restriction
ora_sets <- c(pathways, gmtPathways(file.path(out, "gene_sets_boundary.gmt")))
ora_phyper <- function(u) {
  hits <- intersect(genes, u)
  rows <- lapply(names(ora_sets), function(p) {
    s <- intersect(ora_sets[[p]], u)
    k <- length(intersect(s, hits))
    data.frame(pathway = p, size = length(s), overlap = k,
               pval = phyper(k - 1, length(s), length(u) - length(s), length(hits),
                             lower.tail = FALSE))
  })
  res <- do.call(rbind, rows)
  res <- res[res$size >= min_size & res$size <= max_size, ]
  res$padj <- p.adjust(res$pval, method = "BH")
  attr(res, "N") <- length(u)
  attr(res, "n") <- length(hits)
  res
}

# convention A: the universe exactly as given (what fgsea::fora does)
a <- ora_phyper(universe)
# convention B: the universe intersected with genes annotated to any set in the collection,
# before size filtering (what clusterProfiler::enricher and GOseq-style tools do)
annotated <- unique(unlist(ora_sets))
b <- ora_phyper(intersect(universe, annotated))

ora <- merge(a[, c("pathway", "size", "overlap", "pval", "padj")],
             b[, c("pathway", "size", "overlap", "pval", "padj")],
             by = "pathway", suffixes = c("_given", "_annotated"), all = TRUE)
write_tsv(ora, "ora_phyper.tsv")

fr <- as.data.frame(fora(ora_sets, genes, universe, minSize = min_size, maxSize = max_size))
fr$overlapGenes <- vapply(fr$overlapGenes, paste, "", collapse = ",")
write_tsv(fr[order(fr$pathway), ], "ora_fora.tsv")

cat("ORA convention A (given):     N =", attr(a, "N"), " n =", attr(a, "n"), "\n")
cat("ORA convention B (annotated): N =", attr(b, "N"), " n =", attr(b, "n"), "\n")
m <- merge(fr, a, by = "pathway")
cat("fora vs phyper(given): max |rel diff| pval =",
    max(abs(m$pval.x - m$pval.y) / m$pval.y), " padj =",
    max(abs(m$padj.x - m$padj.y) / m$padj.y), "\n")

# --- preranked GSEA ----------------------------------------------------------------------
stats <- setNames(ranked$stat, ranked$gene)
le <- function(df) {
  df <- as.data.frame(df)
  df$leadingEdge <- vapply(df$leadingEdge, paste, "", collapse = ",")
  df[order(df$pathway), ]
}
set.seed(1)
simple <- fgseaSimple(pathways, stats, nperm = 10000, minSize = min_size, maxSize = max_size,
                      gseaParam = 1, BPPARAM = BiocParallel::SerialParam(progressbar = FALSE))
write_tsv(le(simple), "gsea_fgseaSimple_nperm10000_seed1.tsv")
set.seed(1)
multi <- fgseaMultilevel(pathways, stats, minSize = min_size, maxSize = max_size,
                         gseaParam = 1, eps = 0,
                         BPPARAM = BiocParallel::SerialParam(progressbar = FALSE))
write_tsv(le(multi), "gsea_fgseaMultilevel_eps0_seed1.tsv")

# --- competitive gene-level test (MAGMA-style) -------------------------------------------
# z ~ in_set + log_len over every gene with a statistic; one-sided p for a positive
# membership coefficient
comp <- do.call(rbind, lapply(names(pathways), function(p) {
  in_set <- as.integer(gstats$gene %in% pathways[[p]])
  if (sum(in_set) < min_size || sum(in_set) > max_size) return(NULL)
  fit <- summary(lm(z ~ in_set + log_len, data = gstats))
  co <- fit$coefficients["in_set", ]
  data.frame(pathway = p, size = sum(in_set), beta = co[["Estimate"]], se = co[["Std. Error"]],
             t = co[["t value"]], df = fit$df[2],
             pval = pt(co[["t value"]], fit$df[2], lower.tail = FALSE))
}))
write_tsv(comp, "competitive_lm.tsv")

sessionInfo()
