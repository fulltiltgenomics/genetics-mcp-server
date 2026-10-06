# synthetic, deterministic inputs; the committed files, not this RNG stream, are the fixture
set.seed(20261006)
out <- "/out"

universe <- sprintf("G%04d", 1:900)
# G0801..G0900 are in the universe but in no set, and X001..X100 are in sets but outside the
# universe: the two cases the universe conventions disagree about
annotatable <- c(sprintf("G%04d", 1:800), sprintf("X%03d", 1:100))

# sizes 3 and 400 fall outside the minSize = 5 / maxSize = 300 filter the reference applies
sizes <- c(3, 400, 5, 6, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 50, 60, 70, 80, 100, 120,
           150, 200, 250, 300, 9, 14, 22, 45, 90, 180)
sets <- lapply(sizes, function(s) sort(sample(annotatable, s)))
names(sets) <- sprintf("SET%02d", seq_along(sets))

gmt <- vapply(names(sets), function(n) paste(c(n, "synthetic", sets[[n]]), collapse = "\t"), "")
writeLines(gmt, file.path(out, "gene_sets.gmt"))
writeLines(universe, file.path(out, "universe.txt"))

# enriched for SET10 and SET20, plus genes outside the universe
hits <- c(sample(intersect(sets$SET10, universe), 8),
          sample(intersect(sets$SET20, universe), 15),
          sample(universe, 35),
          sample(sprintf("X%03d", 1:100), 5))
writeLines(sort(unique(hits)), file.path(out, "gene_list.txt"))

# preranked statistic over the universe: SET16 shifted up, SET17 down
stat <- rnorm(length(universe))
names(stat) <- universe
stat[names(stat) %in% sets$SET16] <- stat[names(stat) %in% sets$SET16] + 1.5
stat[names(stat) %in% sets$SET17] <- stat[names(stat) %in% sets$SET17] - 1.0
write.table(data.frame(gene = universe, stat = round(stat, 6)),
            file.path(out, "ranked_stats.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

# gene-level z with a size-like covariate; SET15 members shifted up
log_len <- round(log(runif(length(universe), 1e3, 1e5)), 6)
z <- 0.1 * (log_len - mean(log_len)) + rnorm(length(universe))
z[universe %in% sets$SET15] <- z[universe %in% sets$SET15] + 0.8
write.table(data.frame(gene = universe, z = round(z, 6), log_len = log_len),
            file.path(out, "gene_stats.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

# sets that cross the size filter only through universe restriction, for ORA only: a separate
# file, so the GSEA and competitive outputs over gene_sets.gmt are untouched, and built without
# the RNG so the inputs above are unchanged. SET31/SET32 are 4/5 universe genes plus 3 X*
# (sizes 7/8 as given), SET33/SET34 are 300/301 plus 5 X*. Members come from genes already
# in a set, so the annotated universe is unchanged too
in_sets <- sort(unique(unlist(sets)))
ann_u <- intersect(in_sets, universe)
ann_x <- setdiff(in_sets, universe)
hit_u <- intersect(ann_u, hits)
non_hit <- setdiff(ann_u, hits)
bnd <- list(
  SET31 = c(hit_u[1:2], non_hit[1:2], ann_x[1:3]),
  SET32 = c(hit_u[1:3], non_hit[1:2], ann_x[1:3]),
  SET33 = c(ann_u[1:300], ann_x[1:5]),
  SET34 = c(ann_u[1:301], ann_x[1:5])
)
bnd <- lapply(bnd, sort)
gmt <- vapply(names(bnd), function(n) paste(c(n, "synthetic", bnd[[n]]), collapse = "\t"), "")
writeLines(gmt, file.path(out, "gene_sets_boundary.gmt"))
