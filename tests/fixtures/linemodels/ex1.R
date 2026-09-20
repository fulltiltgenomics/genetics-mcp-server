library(linemodels)
Y = linemodels.ex1[,c("beta1","beta2")]
SE = linemodels.ex1[,c("se1","se2")]
scales = c(0.2, 0.2, 0.2); slopes = c(0, 0.5, 1); cors = c(0.995, 0.995, 0.995)
model.names = c("M0","M.5","M1")
res.lm = line.models(Y, SE, scales, slopes, cors, model.names, model.priors = c(1,1,1), r.lkhood = 0)
write.csv(cbind(linemodels.ex1, res.lm), "/out/ex1_line_models.csv", row.names = FALSE)
print(res.lm[1:3,], digits = 10)
set.seed(1)
res.prop = line.models.with.proportions(Y, SE, scales, slopes, cors, model.names = NULL, r.lkhood = 0, n.iter = 2000, n.burnin = 200, verbose = FALSE)
print(res.prop$params, digits = 6)
par.include = rbind(c(FALSE,FALSE,FALSE), c(TRUE,TRUE,TRUE), c(FALSE,FALSE,FALSE))
op = line.models.optimize(Y, SE, par.include = par.include,
  init.scales = c(0.2, 0.05, 0.2), init.slopes = c(0, 0.2, 1), init.cors = c(0.995, 0.1, 0.995),
  model.priors = c(1,1,1), model.names = model.names, r.lkhood = 0, tol.loglk = 1e-2, op.method = "BFGS", print.steps = 0)
str(op)
print(op$scales, digits=8); print(op$slopes, digits=8); print(op$cors, digits=8)
