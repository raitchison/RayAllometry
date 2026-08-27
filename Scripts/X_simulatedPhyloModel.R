################################################################################
## Worked example: phylogenetic varying-slopes model, on simulated data
##
## WHY THIS IS HERE
##
## The analyses in this repo currently use lm() with a species interaction,
## e.g.  lm(logGSH ~ scaleSPL * Genus_species). That estimates a separate
## slope per species but treats species as independent, and it cannot borrow
## strength across species with few specimens.
##
## This script demonstrates the alternative: a Bayesian varying-slopes model
## in which each species gets its own intercept AND slope, those species
## effects are shrunk toward the overall mean, and their correlation is
## structured by the phylogeny. The 500 trees already prepared in
## Data/Data_Processed/PrunedTrimmedTrees.nex are exactly what such a model
## needs.
##
## It runs on SIMULATED data with known parameter values, so it is fully
## self-contained (no files from this repo are read) and it can be checked:
## section 5 confirms the model recovers the values used to generate the data.
## Nothing here touches the real ray data.
##
## To adapt it to the ray data, the mapping would be roughly:
##   LogGSAcm2      -> log10(GSavg)         (response)
##   LogMeanGSH_cen -> centred log10(SpiracleLength)
##   Binomial       -> Genus_species
##   B              -> vcv.phylo() of a tree from PrunedTrimmedTrees.nex
##
## The model is:
##
##   LogGSAcm2 ~ LogMeanGSH_cen + (1 + LogMeanGSH_cen | gr(Binomial, cov = B))
##
##   i.e. a population-level (fixed) intercept and slope, plus species-level
##   deviations in BOTH intercept and slope, where those species deviations
##   are correlated according to a phylogeny (the covariance matrix B).
##
## The figure shows:
##   - one coloured point per individual, coloured by species
##   - a thick grey line: the population-level fit
##   - thin black lines: each species' own fitted line, clipped to the range
##     of x actually observed for that species
##
## Requires: brms (and a working C++ toolchain / cmdstan), ape, ggplot2.
## Runtime is a few minutes, mostly Stan compilation on the first run.
##
## Author: Nick Dulvy
################################################################################


## ---------------------------------------------------------------------------
## 0. Libraries
## ---------------------------------------------------------------------------

library(here)     # build every file path from the project root
library(ape)      # simulate a phylogeny, build the covariance matrix
library(brms)     # Bayesian multilevel model
library(ggplot2)  # figure

## here() anchors all paths to the top of the project, so the script behaves
## identically whether you run it from RStudio, from Rscript, or with your
## working directory set somewhere unexpected. The root is located via the
## .here file (or the .Rproj file) sitting at the top of this project.
cat("Project root:", here(), "\n")


## ---------------------------------------------------------------------------
## 1. Settings — everything you might want to change lives here
## ---------------------------------------------------------------------------

SEED <- 42

## Data dimensions
N_SPECIES  <- 23    # number of species
MIN_N_OBS  <- 4     # fewest individuals measured for a species
MAX_N_OBS  <- 26    # most individuals measured for a species

## TRUE parameter values used to simulate the data.
## These are set to the posterior means from the real fitted model, so the
## simulated data look like the real thing. The point of the exercise is to
## check that the model can recover them (see section 5).
TRUE_INTERCEPT   <- 10.32   # population-level intercept (log GSA at mean log GSH)
TRUE_SLOPE       <- 2.08    # population-level slope
TRUE_SD_INT      <- 1.73    # among-species SD of intercepts
TRUE_SD_SLOPE    <- 0.26    # among-species SD of slopes
TRUE_COR_INT_SLP <- -0.30   # correlation between species intercept and slope
TRUE_SIGMA       <- 0.20    # residual SD

## Sampler settings.
## These are deliberately light so the demo finishes quickly. The published
## models used warmup = 4000, iter = 16000, adapt_delta = 0.99.
WARMUP        <- 1000
ITER          <- 4000
CHAINS        <- 4
ADAPT_DELTA   <- 0.95
MAX_TREEDEPTH <- 12   # 10 is the default; the phylogenetic term needs more

## Parallel processing. See section 1b below, which works out how many cores
## to use. Leave USE_PARALLEL = TRUE unless you are debugging.
USE_PARALLEL  <- TRUE
RESERVE_CORES <- 1     # leave this many cores free so the machine stays usable
USE_THREADING <- TRUE  # also split each chain across cores (needs cmdstanr)

## Where output goes. All paths are built with here(), so they resolve from
## the project root rather than from your current working directory.
DIR_PLOTS  <- here("Results", "Figures")
DIR_MODELS <- here("Results", "Models")

SAVE_PLOTS <- TRUE    # write the figures to Results/Figures/
SAVE_MODEL <- TRUE    # write the fitted brms object to Results/Models/

## Create the output folders if they are not already there. Git does not track
## empty directories, so anyone cloning this project gets them made on the fly.
dir.create(DIR_PLOTS,  recursive = TRUE, showWarnings = FALSE)
dir.create(DIR_MODELS, recursive = TRUE, showWarnings = FALSE)

set.seed(SEED)


## ---------------------------------------------------------------------------
## 1b. Parallel processing — works on macOS, Windows and Linux
## ---------------------------------------------------------------------------
##
## Two different kinds of parallelism are available, and they stack:
##
##   1. BETWEEN chains. Each MCMC chain runs on its own core. This is the easy
##      win, but it caps out at CHAINS cores (4 here) no matter how big your
##      machine is.
##
##   2. WITHIN a chain ("threading"). The likelihood for one chain is split
##      across several cores. This is what lets a 10-core machine use more than
##      4 cores. It needs the cmdstanr backend; if that is not installed the
##      script quietly falls back to chain-level parallelism only.
##
## Platform notes:
##   macOS / Linux - R forks the session; nothing extra to install.
##   Windows       - R cannot fork, so rstan starts background worker
##                   processes instead. This works fine, it just takes a few
##                   extra seconds to spin up. No code change is needed.

os_name  <- Sys.info()[["sysname"]]
os_label <- switch(os_name,
                   Darwin  = "macOS",
                   Windows = "Windows",
                   Linux   = "Linux",
                   os_name)

## How many cores can we actually use?
## parallelly::availableCores() is the most reliable, because it respects
## limits set by shared servers, HPC schedulers and CI runners. We use it when
## it is installed and fall back to base R otherwise.
detect_cores_safe <- function() {
  if (requireNamespace("parallelly", quietly = TRUE)) {
    n <- parallelly::availableCores()
  } else {
    ## physical cores are the better guide for Stan; hyperthreads help little
    n <- parallel::detectCores(logical = FALSE)
    if (is.na(n) || n < 1L) n <- parallel::detectCores(logical = TRUE)
  }
  if (is.na(n) || n < 1L) n <- 1L   # some Windows / container setups return NA
  as.integer(n)
}

N_CORES_TOTAL <- detect_cores_safe()

## Use all but RESERVE_CORES, never fewer than 1
N_CORES_USABLE <- if (USE_PARALLEL) {
  max(1L, N_CORES_TOTAL - RESERVE_CORES)
} else {
  1L
}

## Chain-level parallelism cannot use more cores than there are chains
CORES <- min(CHAINS, N_CORES_USABLE)

## Any cores left over can go to within-chain threading
THREADS_PER_CHAIN <- max(1L, N_CORES_USABLE %/% CORES)

## Threading only works with the cmdstanr backend
HAVE_CMDSTANR <- requireNamespace("cmdstanr", quietly = TRUE)
DO_THREADING  <- USE_THREADING && HAVE_CMDSTANR && THREADS_PER_CHAIN > 1L

## rstan and several other packages read this option
options(mc.cores = CORES)

cat("\n---- Parallel settings ----\n")
cat("Platform            :", os_label, "\n")
cat("Cores detected      :", N_CORES_TOTAL, "\n")
cat("Cores reserved      :", RESERVE_CORES, "\n")
cat("Cores in use        :", N_CORES_USABLE, "\n")
cat("Chains              :", CHAINS, "(run on", CORES, "cores)\n")
if (DO_THREADING) {
  cat("Within-chain threads:", THREADS_PER_CHAIN,
      "per chain (cmdstanr backend)\n")
  cat("Total cores busy    :", CORES * THREADS_PER_CHAIN, "\n")
} else if (USE_THREADING && !HAVE_CMDSTANR) {
  cat("Within-chain threads: off - cmdstanr is not installed.\n")
  cat("                      Chains still run in parallel on", CORES, "cores.\n")
  cat("                      To use the spare cores, install cmdstanr:\n")
  cat("                      install.packages('cmdstanr',\n")
  cat("                        repos = c('https://mc-stan.org/r-packages/',\n")
  cat("                                  getOption('repos')))\n")
  cat("                      then: cmdstanr::install_cmdstan()\n")
} else {
  cat("Within-chain threads: off\n")
}
cat("---------------------------\n\n")


## ---------------------------------------------------------------------------
## 2. Simulate a phylogeny and the species-level effects
## ---------------------------------------------------------------------------

## 2.1 A random ultrametric tree standing in for the shark phylogeny.
tree <- rcoal(N_SPECIES)
tree$tip.label <- paste0("Species_", sprintf("%02d", seq_len(N_SPECIES)))

## 2.2 The phylogenetic correlation matrix.
## corr = TRUE rescales to a correlation matrix, which is what brms expects
## for the `cov` argument of gr(). Closely related species have entries near 1.
B <- vcv.phylo(tree, corr = TRUE)

## 2.3 Species deviations in intercept and slope.
##
## brms's (1 + x | gr(sp, cov = B)) implies the stacked species effects are
##   vec(U) ~ MVN(0, Sigma %x% B)
## where Sigma is the 2x2 intercept/slope covariance matrix. We can draw that
## directly: if Z is N_SPECIES x 2 of iid normals, A = t(chol(B)) and
## C = chol(Sigma), then U = A %*% Z %*% C has exactly that covariance.

Sigma <- matrix(
  c(TRUE_SD_INT^2,                                    # var(intercept)
    TRUE_COR_INT_SLP * TRUE_SD_INT * TRUE_SD_SLOPE,   # cov(intercept, slope)
    TRUE_COR_INT_SLP * TRUE_SD_INT * TRUE_SD_SLOPE,
    TRUE_SD_SLOPE^2),                                 # var(slope)
  nrow = 2,
  dimnames = list(c("Intercept", "Slope"), c("Intercept", "Slope"))
)

Z <- matrix(rnorm(N_SPECIES * 2), nrow = N_SPECIES, ncol = 2)
U <- t(chol(B)) %*% Z %*% chol(Sigma)

species_effects <- data.frame(
  Binomial  = tree$tip.label,
  u_int     = U[, 1],
  u_slope   = U[, 2],
  row.names = NULL
)


## ---------------------------------------------------------------------------
## 3. Simulate the observations
## ---------------------------------------------------------------------------

## Each species is measured over its own limited body-size range, so species
## occupy different, partly overlapping windows on the x axis. That is what
## makes the within- vs among-species distinction interesting in the first
## place, so we simulate it explicitly rather than drawing x from one
## common distribution.

sim_one_species <- function(i) {
  sp <- species_effects$Binomial[i]

  n_obs <- sample(MIN_N_OBS:MAX_N_OBS, 1)

  ## this species' own window of log gill slit height
  sp_centre <- rnorm(1, mean = 0, sd = 0.9)   # where the species sits on x
  sp_spread <- runif(1, min = 0.15, max = 0.6) # how wide its size range is

  x_raw <- rnorm(n_obs, mean = sp_centre, sd = sp_spread)

  data.frame(
    Binomial       = sp,
    LogMeanGSH_raw = x_raw,
    u_int          = species_effects$u_int[i],
    u_slope        = species_effects$u_slope[i]
  )
}

sim_df <- do.call(rbind, lapply(seq_len(N_SPECIES), sim_one_species))

## Centre the predictor across the whole data set, exactly as in the real
## analysis. The "_cen" suffix is what makes the intercept interpretable as
## log GSA at the mean log GSH rather than at log GSH = 0.
sim_df$LogMeanGSH_cen <- as.numeric(scale(sim_df$LogMeanGSH_raw, center = TRUE, scale = FALSE))

## The response: population effects + species deviations + residual noise
sim_df$LogGSAcm2 <-
  (TRUE_INTERCEPT + sim_df$u_int) +
  (TRUE_SLOPE     + sim_df$u_slope) * sim_df$LogMeanGSH_cen +
  rnorm(nrow(sim_df), mean = 0, sd = TRUE_SIGMA)

## Drop the columns the model is not allowed to see
sim_df <- sim_df[, c("Binomial", "LogMeanGSH_cen", "LogGSAcm2")]

cat("\nSimulated", nrow(sim_df), "observations across",
    length(unique(sim_df$Binomial)), "species\n")
print(summary(as.numeric(table(sim_df$Binomial))))


## ---------------------------------------------------------------------------
## 4. Fit the model
## ---------------------------------------------------------------------------

## get_prior() shows the default (weakly informative) priors brms will use.
## We pass them back in explicitly so the choice is visible in the script
## rather than implicit.
fit_prior <- get_prior(
  LogGSAcm2 ~ LogMeanGSH_cen + (1 + LogMeanGSH_cen | gr(Binomial, cov = B)),
  data   = sim_df,
  data2  = list(B = B),
  family = gaussian()
)

print(fit_prior)

## The arguments are assembled in a list so the threading options can be added
## only when cmdstanr is available (see section 1b). Everything else is the
## same either way.
brm_args <- list(
  formula = LogGSAcm2 ~ LogMeanGSH_cen + (1 + LogMeanGSH_cen | gr(Binomial, cov = B)),
  data    = sim_df,
  data2   = list(B = B),        # <- the phylogenetic covariance matrix
  family  = gaussian(),
  prior   = fit_prior,
  warmup  = WARMUP,
  iter    = ITER,
  chains  = CHAINS,
  cores   = CORES,              # <- chains run in parallel across cores
  control = list(adapt_delta = ADAPT_DELTA, max_treedepth = MAX_TREEDEPTH),
  seed    = SEED
)

if (DO_THREADING) {
  brm_args$backend   <- "cmdstanr"
  brm_args$threading <- threading(THREADS_PER_CHAIN)
}

fit <- do.call(brm, brm_args)

print(summary(fit))

## Save the fitted object so you can come back to it without refitting.
## Reload later with:  fit <- readRDS(here("Results", "Models", "simulatedPhyloFit.RDS"))
if (SAVE_MODEL) {
  saveRDS(fit, here("Results", "Models", "simulatedPhyloFit.RDS"))
  cat("\nModel written to:", here("Results", "Models", "simulatedPhyloFit.RDS"), "\n")
}


## ---------------------------------------------------------------------------
## 5. Did we get the truth back?
## ---------------------------------------------------------------------------

vc <- VarCorr(fit)

recovery <- data.frame(
  parameter = c("Intercept", "Slope", "sd(Intercept)", "sd(Slope)",
                "cor(Int,Slope)", "sigma"),
  truth     = c(TRUE_INTERCEPT, TRUE_SLOPE, TRUE_SD_INT, TRUE_SD_SLOPE,
                TRUE_COR_INT_SLP, TRUE_SIGMA),
  estimate  = c(fixef(fit)["Intercept", "Estimate"],
                fixef(fit)["LogMeanGSH_cen", "Estimate"],
                vc$Binomial$sd["Intercept", "Estimate"],
                vc$Binomial$sd["LogMeanGSH_cen", "Estimate"],
                vc$Binomial$cor["Intercept", "Estimate", "LogMeanGSH_cen"],
                vc$residual__$sd[, "Estimate"]),
  q2.5      = c(fixef(fit)["Intercept", "Q2.5"],
                fixef(fit)["LogMeanGSH_cen", "Q2.5"],
                vc$Binomial$sd["Intercept", "Q2.5"],
                vc$Binomial$sd["LogMeanGSH_cen", "Q2.5"],
                vc$Binomial$cor["Intercept", "Q2.5", "LogMeanGSH_cen"],
                vc$residual__$sd[, "Q2.5"]),
  q97.5     = c(fixef(fit)["Intercept", "Q97.5"],
                fixef(fit)["LogMeanGSH_cen", "Q97.5"],
                vc$Binomial$sd["Intercept", "Q97.5"],
                vc$Binomial$sd["LogMeanGSH_cen", "Q97.5"],
                vc$Binomial$cor["Intercept", "Q97.5", "LogMeanGSH_cen"],
                vc$residual__$sd[, "Q97.5"])
)

recovery$covered <- with(recovery, truth >= q2.5 & truth <= q97.5)

cat("\n---- Parameter recovery (is the truth inside the 95% CI?) ----\n")
print(recovery, digits = 3, row.names = FALSE)


## ---------------------------------------------------------------------------
## 6. Species-level fitted lines
## ---------------------------------------------------------------------------

## ranef() returns species DEVIATIONS from the population-level effects, so
## they are centred near zero. To draw each species' actual line we add the
## population-level (fixed) intercept and slope back on.
##
## Note: ranef() for a brmsfit takes (object, summary, robust, probs, pars,
## groups). Arguments such as `var` or `center.zero` are lme4/nlme arguments;
## brms silently swallows them via `...`, so they do nothing here.

rf <- as.data.frame(ranef(fit)$Binomial)
rf$Binomial <- rownames(rf)

f_int <- fixef(fit)["Intercept", "Estimate"]
f_slp <- fixef(fit)["LogMeanGSH_cen", "Estimate"]

rf$Estimate.Intercept      <- rf$Estimate.Intercept      + f_int
rf$Estimate.LogMeanGSH_cen <- rf$Estimate.LogMeanGSH_cen + f_slp

cat("\nRange of species-level slopes:",
    paste(round(range(rf$Estimate.LogMeanGSH_cen), 3), collapse = " to "),
    "  (population-level slope:", round(f_slp, 3), ")\n")

## Build one line segment per species, clipped to the x range that species
## was actually measured over. Clipping matters: an unclipped line implies
## you have data on a species where you do not.
species_lines <- do.call(rbind, lapply(unique(sim_df$Binomial), function(sp) {
  xr <- range(sim_df$LogMeanGSH_cen[sim_df$Binomial == sp])
  a  <- rf$Estimate.Intercept[rf$Binomial == sp]
  b  <- rf$Estimate.LogMeanGSH_cen[rf$Binomial == sp]
  data.frame(Binomial = sp, x = xr[1], xend = xr[2],
             y = a + b * xr[1], yend = a + b * xr[2])
}))

## The population-level line, spanning the full x range
x_all      <- range(sim_df$LogMeanGSH_cen)
pop_line   <- data.frame(
  x    = x_all[1], xend = x_all[2],
  y    = f_int + f_slp * x_all[1],
  yend = f_int + f_slp * x_all[2]
)


## ---------------------------------------------------------------------------
## 7. The figure
## ---------------------------------------------------------------------------

## annotate() / a separate data frame draws each segment ONCE. (Passing x, y,
## xend, yend as fixed parameters to geom_segment() while it inherits the main
## data frame silently redraws the same segment once per row, which is why
## semi-transparent segments came out looking opaque in earlier versions.)

Plot_within <- ggplot(sim_df, aes(x = LogMeanGSH_cen, y = LogGSAcm2,
                                  colour = Binomial)) +
  ## the data
  geom_point(size = 3, alpha = 0.7) +
  ## species-level lines, clipped to each species' observed range.
  ## colour is mapped to Binomial, so each line takes the colour of its own
  ## species' points. Both layers draw from the same discrete scale, so the
  ## match is exact rather than something we have to line up by hand.
  geom_segment(data = species_lines,
               aes(x = x, xend = xend, y = y, yend = yend,
                   colour = Binomial),
               linewidth = 1,
               inherit.aes = FALSE) +
  ## population-level line LAST, so it sits on top of the species lines
  ## and the points rather than being buried under them
  annotate("segment",
           x = pop_line$x, xend = pop_line$xend,
           y = pop_line$y, yend = pop_line$yend,
           colour = "grey70", linewidth = 2) +
  labs(x = "LogMeanGSH_cen", y = "LogGSAcm2") +
  theme_bw() +
  theme(legend.position = "none")

print(Plot_within)

if (SAVE_PLOTS) {
  ggsave(here("Results", "Figures", "simulatedWithinSpecies.png"),
         Plot_within, height = 6, width = 6, units = "in", dpi = 300)
  cat("\nFigure written to:",
      here("Results", "Figures", "simulatedWithinSpecies.png"), "\n")
}


## ---------------------------------------------------------------------------
## 8. Companion figure: population-level line only ("across species")
## ---------------------------------------------------------------------------

Plot_across <- ggplot(sim_df, aes(x = LogMeanGSH_cen, y = LogGSAcm2,
                                  colour = Binomial)) +
  annotate("segment",
           x = pop_line$x, xend = pop_line$xend,
           y = pop_line$y, yend = pop_line$yend,
           colour = "grey30", linewidth = 2) +
  geom_point(size = 3, alpha = 0.7) +
  labs(x = "LogMeanGSH_cen", y = "LogGSAcm2") +
  theme_bw() +
  theme(legend.position = "none")

print(Plot_across)

if (SAVE_PLOTS) {
  ggsave(here("Results", "Figures", "simulatedAcrossSpecies.png"),
         Plot_across, height = 6, width = 6, units = "in", dpi = 300)
  cat("Figure written to:",
      here("Results", "Figures", "simulatedAcrossSpecies.png"), "\n")
}

################################################################################
## End
################################################################################
