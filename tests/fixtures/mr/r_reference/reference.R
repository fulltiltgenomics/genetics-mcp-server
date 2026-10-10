# Reference outputs for genetics_mcp_server.sdk.mr from MendelianRandomization and TwoSampleMR.
# Run inside the mr-r-reference image with this directory mounted at /out (README.md).
suppressPackageStartupMessages({
  library(MendelianRandomization)
  library(TwoSampleMR)
})

DIR <- if (nzchar(Sys.getenv("OUT"))) Sys.getenv("OUT") else "/out"
SEED <- 20261010L          # set.seed() before every TwoSampleMR call that bootstraps
MR_SEED <- 314159265L      # MendelianRandomization's own default, passed explicitly
MR_ITER <- 10000L
PREVALENCES <- c(0.1, 0.005)  # 0.1 is TwoSampleMR's assumed default; 0.005 is closer to IBD

num <- function(x) ifelse(is.na(x), "NA", sprintf("%.17g", as.numeric(x)))
est <- list(); pv <- list()
put <- function(table, package, method, options, quantity, value) {
  est[[length(est) + 1]] <<- data.frame(table = table, package = package, method = method,
    options = options, quantity = quantity,
    value = if (is.numeric(value)) num(value) else as.character(value))
}
put_pv <- function(table, package, method, options, variant, quantity, value) {
  pv[[length(pv) + 1]] <<- data.frame(table = table, package = package, method = method,
    options = options, variant = variant, quantity = quantity,
    value = if (is.numeric(value)) num(value) else as.character(value))
}
guard <- function(table, package, method, options, expr) {
  tryCatch(expr, error = function(e) put(table, package, method, options, "error", conditionMessage(e)))
}
slots <- function(table, method, options, obj, names) {
  if (is.null(obj)) stop("the package returned NULL (it refuses this number of variants)")
  for (n in names) {
    v <- slot(obj, n)
    if (n == "Heter.Stat") { put(table, "MendelianRandomization", method, options, "Q", v[1])
                             put(table, "MendelianRandomization", method, options, "Q_pval", v[2]) }
    else put(table, "MendelianRandomization", method, options, n, v)
  }
}
IVW_SLOTS <- c("Model", "Estimate", "StdError", "CILower", "CIUpper", "Pvalue", "Heter.Stat", "RSE", "SNPs")
EGGER_SLOTS <- c("Model", "Estimate", "StdError.Est", "CILower.Est", "CIUpper.Est", "Pvalue.Est",
                 "Intercept", "StdError.Int", "CILower.Int", "CIUpper.Int", "Pvalue.Int",
                 "Heter.Stat", "I.sq", "RSE", "SNPs")

run_mr <- function(name, d, R) {
  K <- nrow(d)
  # Wald ratio, first-order SE: no package function, so stated here
  for (i in seq_len(K)) {
    put_pv(name, "base", "wald", "first_order", d$variant[i], "estimate", d$beta_y[i] / d$beta_x[i])
    put_pv(name, "base", "wald", "first_order", d$variant[i], "se", d$se_y[i] / abs(d$beta_x[i]))
  }
  inp <- MendelianRandomization::mr_input(bx = d$beta_x, bxse = d$se_x, by = d$beta_y, byse = d$se_y, snps = d$variant)
  for (model in c("default", "fixed", "random")) {
    opt <- paste0("model=", model)
    guard(name, "MendelianRandomization", "mr_ivw", opt, {
      o <- MendelianRandomization::mr_ivw(inp, model = model)
      slots(name, "mr_ivw", opt, o, IVW_SLOTS)
      Q <- o@Heter.Stat[1]
      # I2 is not a slot; the usual (Q - df) / Q floored at 0 on mr_ivw's Q
      put(name, "MendelianRandomization", "mr_ivw", opt, "I2_from_Q", max(0, (Q - (K - 1)) / Q))
    })
  }
  if (!is.null(R)) {
    keep <- match(colnames(R), d$variant)
    dc <- d[keep, ]
    inpc <- MendelianRandomization::mr_input(bx = dc$beta_x, bxse = dc$se_x, by = dc$beta_y, byse = dc$se_y,
                     snps = dc$variant, correlation = R)
    for (model in c("fixed", "random")) {
      opt <- paste0("correl=TRUE;model=", model)
      guard(name, "MendelianRandomization", "mr_ivw", opt,
            slots(name, "mr_ivw", opt, MendelianRandomization::mr_ivw(inpc, model = model, correl = TRUE), IVW_SLOTS))
    }
    guard(name, "MendelianRandomization", "mr_egger", "correl=TRUE",
          slots(name, "mr_egger", "correl=TRUE", MendelianRandomization::mr_egger(inpc, correl = TRUE), EGGER_SLOTS))
  }
  guard(name, "MendelianRandomization", "mr_egger", "default",
        slots(name, "mr_egger", "default", MendelianRandomization::mr_egger(inp), EGGER_SLOTS))
  for (w in c("weighted", "simple")) {
    opt <- sprintf("weighting=%s;iterations=%d;seed=%d", w, MR_ITER, MR_SEED)
    guard(name, "MendelianRandomization", "mr_median", opt,
          slots(name, "mr_median", opt,
                MendelianRandomization::mr_median(inp, weighting = w, iterations = MR_ITER, seed = MR_SEED),
                c("Estimate", "StdError", "CILower", "CIUpper", "Pvalue", "SNPs")))
  }
  for (w in c("weighted", "unweighted")) {
    opt <- sprintf("weighting=%s;stderror=delta;phi=1;iterations=%d;seed=%d", w, MR_ITER, MR_SEED)
    guard(name, "MendelianRandomization", "mr_mbe", opt,
          slots(name, "mr_mbe", opt,
                MendelianRandomization::mr_mbe(inp, weighting = w, stderror = "delta", phi = 1, iterations = MR_ITER, seed = MR_SEED),
                c("Estimate", "StdError", "CILower", "CIUpper", "Pvalue", "SNPs")))
  }
  run_tsmr(name, d)
}

tsmr_frame <- function(name, d, binary, prevalence = NA) {
  dat <- data.frame(SNP = d$variant, exposure = name, id.exposure = name,
    outcome = if (binary) "IBD" else "synthetic outcome", id.outcome = "out",
    beta.exposure = d$beta_x, se.exposure = d$se_x, eaf.exposure = d$af_x,
    samplesize.exposure = d$n_x, pval.exposure = 2 * pnorm(-abs(d$beta_x / d$se_x)),
    beta.outcome = d$beta_y, se.outcome = d$se_y, eaf.outcome = d$af_y,
    samplesize.outcome = d$n_y, pval.outcome = 2 * pnorm(-abs(d$beta_y / d$se_y)),
    units.exposure = "SD", units.outcome = if (binary) "log odds" else "SD",
    mr_keep = TRUE, stringsAsFactors = FALSE)
  if (binary) {
    dat$ncase.outcome <- d$ncase_y
    dat$ncontrol.outcome <- d$ncontrol_y
    dat$prevalence.outcome <- prevalence
  }
  dat
}

run_tsmr <- function(name, d) {
  binary <- "ncase_y" %in% names(d)
  dat <- tsmr_frame(name, d, binary, PREVALENCES[1])
  P <- "TwoSampleMR"
  dp <- default_parameters()
  opt <- sprintf("defaults;seed=%d;nboot=%d;phi=%s", SEED, dp$nboot, dp$phi)
  guard(name, P, "mr", opt, {
    set.seed(SEED)
    m <- suppressMessages(mr(dat))
    for (i in seq_len(nrow(m))) for (q in c("nsnp", "b", "se", "pval"))
      put(name, P, paste0("mr:", m$method[i]), opt, q, m[[q]][i])
  })
  guard(name, P, "mr_heterogeneity", "defaults", {
    h <- mr_heterogeneity(dat)
    for (i in seq_len(nrow(h))) for (q in c("Q", "Q_df", "Q_pval"))
      put(name, P, paste0("mr_heterogeneity:", h$method[i]), "defaults", q, h[[q]][i])
  })
  guard(name, P, "mr_pleiotropy_test", "defaults", {
    p <- mr_pleiotropy_test(dat)
    for (q in c("egger_intercept", "se", "pval")) put(name, P, "mr_pleiotropy_test", "defaults", q, p[[q]])
  })
  guard(name, P, "mr_singlesnp", "defaults", {
    s <- mr_singlesnp(dat)
    for (i in seq_len(nrow(s))) for (q in c("b", "se", "p"))
      put_pv(name, P, "mr_singlesnp", "defaults", s$SNP[i], q, s[[q]][i])
  })
  guard(name, P, "mr_leaveoneout", "defaults", {
    l <- mr_leaveoneout(dat)
    for (i in seq_len(nrow(l))) for (q in c("b", "se", "p"))
      put_pv(name, P, "mr_leaveoneout", "defaults", l$SNP[i], q, l[[q]][i])
  })
  for (prev in if (binary) PREVALENCES else NA) {
    dat <- tsmr_frame(name, d, binary, prev)
    opt <- if (binary) sprintf("units.outcome=log odds;prevalence=%s", prev) else "units=SD"
    # the r each convention gives, per variant, so a port can target the functions directly
    r_exp <- get_r_from_bsen(dat$beta.exposure, dat$se.exposure, dat$samplesize.exposure)
    r_out <- if (binary) get_r_from_lor(dat$beta.outcome, dat$eaf.outcome, dat$ncase.outcome,
                                        dat$ncontrol.outcome, prev)
             else get_r_from_bsen(dat$beta.outcome, dat$se.outcome, dat$samplesize.outcome)
    for (i in seq_len(nrow(dat))) {
      put_pv(name, P, "get_r_from_bsen:exposure", opt, dat$SNP[i], "r", r_exp[i])
      put_pv(name, P, if (binary) "get_r_from_lor:outcome" else "get_r_from_bsen:outcome",
             opt, dat$SNP[i], "r", r_out[i])
      if (binary) {
        put_pv(name, P, "input", opt, dat$SNP[i], "ncase", dat$ncase.outcome[i])
        put_pv(name, P, "input", opt, dat$SNP[i], "ncontrol", dat$ncontrol.outcome[i])
      }
    }
    guard(name, P, "steiger_filtering", opt, {
      s <- suppressWarnings(suppressMessages(steiger_filtering(dat)))
      for (i in seq_len(nrow(s))) for (q in c("rsq.exposure", "rsq.outcome", "steiger_dir", "steiger_pval"))
        put_pv(name, P, "steiger_filtering", opt, s$SNP[i], q,
               if (is.logical(s[[q]])) as.character(s[[q]][i]) else s[[q]][i])
    })
    guard(name, P, "directionality_test", opt, {
      t <- suppressWarnings(suppressMessages(directionality_test(dat)))
      for (q in c("snp_r2.exposure", "snp_r2.outcome", "correct_causal_direction", "steiger_pval"))
        put(name, P, "directionality_test", opt, q,
            if (is.logical(t[[q]])) as.character(t[[q]]) else t[[q]])
    })
    guard(name, P, "mr_steiger", opt, {
      ms <- mr_steiger(dat$pval.exposure, dat$pval.outcome, dat$samplesize.exposure,
                       dat$samplesize.outcome, r_exp, r_out)
      for (q in c("r2_exp", "r2_out", "correct_causal_direction", "steiger_test"))
        put(name, P, "mr_steiger", opt, q,
            if (is.logical(ms[[q]])) as.character(ms[[q]]) else ms[[q]])
    })
  }
}

cis <- read.csv(file.path(DIR, "input_cis_ibd.csv"), stringsAsFactors = FALSE)
for (p in sort(unique(cis$protein))) {
  R <- as.matrix(read.csv(file.path(DIR, sprintf("ld_%s.csv", p)), row.names = 1, check.names = FALSE))
  run_mr(p, cis[cis$protein == p, ], R)
}
run_mr("synthetic", read.csv(file.path(DIR, "input_synthetic.csv"), stringsAsFactors = FALSE), NULL)

write.csv(do.call(rbind, est), file.path(DIR, "out_estimates.csv"), row.names = FALSE)
write.csv(do.call(rbind, pv), file.path(DIR, "out_per_variant.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(DIR, "sessionInfo.txt"))
