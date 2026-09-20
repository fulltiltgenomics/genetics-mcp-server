library(linemodels)
write.csv(linemodels.ex1, "/out/linemodels_ex1.csv", row.names = FALSE)
write.csv(linemodels.ex2, "/out/linemodels_ex2.csv", row.names = FALSE)
# --- Example 2: COVID-19 HGI r6 B2 vs C2
data.file = "https://raw.githubusercontent.com/mjpirinen/covid19-hgi_subtypes/main/covid_hgi_v6_B2_C2_common.tsv"
dat = read.csv(data.file, sep = "\t", header = TRUE, as.is = TRUE)
write.table(dat, "/out/covid_hgi_v6_B2_C2_common.tsv", sep = "\t", row.names = FALSE, quote = FALSE)
dat = dat[dat[,"SNP"] != "12:112914354:T:C",]
X = dat[,c("B2_beta","C2_beta")]; SE = dat[,c("B2_sebeta","C2_sebeta")]
ii = X[,1] < 0; X[ii,] = -X[ii,]
slope.both = tan(atan(0.2) + (atan(1) - atan(0.2))/2)
cat("slope.both", format(slope.both, digits = 12), "\n")
scales = c(0.15, 0.15, 0.15); slopes = c(0.2, 1, slope.both); cors = c(0.999, 0.999, 0.999)
model.names = c("SEVER.", "SUSCEP.", "BOTH"); r.lkhood = 0.4539485
res.1 = line.models(X, SE, scales, slopes, cors, model.names, model.priors = rep(1/3, 3), r.lkhood = r.lkhood, scale.weights = c(1))
write.csv(cbind(SNP = dat$SNP, X, SE, res.1), "/out/covid_line_models.csv", row.names = FALSE)
set.seed(91)
res.2 = line.models.with.proportions(X, SE, scales, slopes, cors, model.names, r.lkhood = r.lkhood, n.iter = 10000, n.burnin = 50, verbose = FALSE)
print(res.2$params, digits = 6)
print(colSums(res.2$groups > 0.95)); print(nrow(res.2$groups))
write.csv(cbind(SNP = dat$SNP, res.2$groups), "/out/covid_proportions_groups.csv", row.names = FALSE)
write.csv(res.2$params, "/out/covid_proportions_params.csv")
# --- Example 4: ex2 2D, without features
X = linemodels.ex2[,c("beta1","beta2")]; SE = linemodels.ex2[,c("se1","se2")]
slopes = matrix(c(0, 3), ncol = 1); scales = rep(0.1, 2); cors = rep(0.999, 2)
res.fix = line.models(X, SE, scales, slopes, cors, r.lkhood = 0)
write.csv(res.fix, "/out/ex2_line_models.csv", row.names = FALSE)
set.seed(7)
res.p = line.models.with.proportions(X, SE, scales, slopes, cors, model.names = NULL, r.lkhood = 0, n.iter = 2000, n.burnin = 200, verbose = FALSE)
print(res.p$params, digits = 6); print(res.p$groups[101,], digits = 6)
set.seed(7)
res.f = line.models.with.features(X, SE, scales, slopes, cors, model.names = NULL, r.lkhood = 0, n.iter = 2000, n.burnin = 200, features = linemodels.ex2$annotation, verbose = FALSE)
print(res.f$params, digits = 6); print(res.f$groups[101,], digits = 6)
# --- Example 3: 3D optimize with constant SE
sc = sqrt(2 * linemodels.ex1$maf * (1 - linemodels.ex1$maf))
Y.sc = linemodels.ex1[,c("beta1","beta2","beta3")]*sc; SE.sc = linemodels.ex1[,c("se1","se2","se3")]*sc
par.include = list(scales = c(TRUE,TRUE,TRUE), slopes = matrix(TRUE, ncol = 2, nrow = 3), cors = c(FALSE,FALSE,FALSE))
op.res = line.models.optimize(Y.sc, SE.sc, par.include = par.include, init.scales = rep(0.15, 3),
  init.slopes = matrix(c(0,0, 1,1, 0.5,0.5), byrow = TRUE, ncol = 2), init.cors = c(0.995,0.995,0.995),
  force.same.scales = TRUE, r.lkhood = c(0,0,0), tol.loglk = 1e-2, op.method = "BFGS", assume.constant.SE = TRUE, print.steps = 0)
print(op.res, digits = 8)
# 2D scaled optimize from the examples file (expected scales 0.1515844 0.1418315 0.1350342)
Y2 = linemodels.ex1[,c("beta1","beta2")]*sc; SE2 = linemodels.ex1[,c("se1","se2")]*sc
op2 = line.models.optimize(Y2, SE2, par.include = rbind(c(TRUE,FALSE,FALSE), c(TRUE,TRUE,TRUE), c(TRUE,FALSE,FALSE)),
  init.scales = c(1,1,1), init.slopes = c(0, 0.2, 1), init.cors = c(0.995, 0.1, 0.995), model.priors = c(1,1,1),
  r.lkhood = 0, tol.loglk = 1e-2, assume.constant.SE = TRUE, op.method = "BFGS", print.steps = 0)
print(op2, digits = 8)
