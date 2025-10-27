library(pacman)
p_load(dplyr, lubridate, ggplot2,brms,tidyr)

f_decay <- function(values, k_d, t_d){
    1/ (1 + exp(k_d *(values - t_d)))
}

f_flush <- function(values, k_f, t_f){
    1/ (1 + exp(-k_f * (values - t_f)))
}

values1 <- seq(0, 365, by=1)

LC <- function(values, k_d, k_f, t_d, t_f) {
  result <- (f_decay(values, k_d, t_d) +
        f_flush(values, k_f, t_f))
  return(result)
}

LC_values<- LC(values1, k_d=0.8, k_f=0.9, t_d=68, t_f=43)

windows()
plot(values1, LC_values, type='l', col='black', ylim=c(0,10))


data<- read.csv("timeseries\\simulated_phenophase_data.csv")
data <- data %>%
  mutate(
    y_norm= pmin(pmax(observed_leafing / 100, 1e-4), 1 - 1e-4),
    date = as.Date(date),
    date_num = as.numeric(difftime(date, as.Date("2018-04-04"), units = "days")),
    DOY= yday(date),
    year= year(date),
    month= month(date),
    pheno_year = if_else(month >= 9, year, year - 1),
    day = as.numeric(difftime(date, as.Date(paste0(pheno_year, "-09-01")), units = "days")),
    tree= as.factor(tree),
    pheno_year= as.factor(pheno_year),
    tree_year= as.factor(paste0(tree, "_", pheno_year))
  )



windows()
ggplot(data, aes(x=day, y=y_norm, color=tree_year)) +
  geom_line() +
  facet_wrap(~pheno_year,ncol=1)+
  scale_x_continuous(breaks=seq(0,365,30), limits=c(0,365))


trees<- unique(data$tree)
years<- unique(data$pheno_year)

all_before_threshold <- data.frame()
for (i in 1:length(trees)) {
  for (j in 1:length(years)) {
    subset_data <- data %>% filter(tree == trees[i], pheno_year == years[j])
    windows()
    plot(subset_data$day, subset_data$y_norm, type='l', main=paste("Tree:", trees[i], "Year:", years[j]),
         xlab="Day", ylab="Normalized Leafing")

    #find values below threshold
    threshold <- 0.2
    below_threshold <- which(subset_data$y_norm < threshold)
    last_item<-max(below_threshold)
    #subset all values before last_item
    if (length(below_threshold) > 0 && last_item < nrow(subset_data)) {
      subset_before_threshold <- subset_data[1:last_item, ]
      points(subset_before_threshold$day, subset_before_threshold$y_norm, col='red', pch=19)
    }
    ##concat all subset_before_threshold into a new dataframe
    all_before_threshold <- bind_rows(all_before_threshold, subset_before_threshold)
}

all_before_threshold<-all_before_threshold %>%
  filter(tree %in% c('A','B','C','D','E'))

windows()
ggplot(all_before_threshold, aes(x=day, y=y_norm, color=tree_year)) +
  geom_line() +
  facet_wrap(~pheno_year,ncol=1)+
  scale_x_continuous(breaks=seq(0,365,30), limits=c(0,365))



form_main <- bf(
  y_norm ~ (1 / (1 + exp(kd * (day - td)))),
  kd + td ~ 1 + pheno_year + (1 | tree),
  nl = TRUE
)


priors<-c (
    prior(normal(170, 30), nlpar = "td", lb = 0),
    prior(normal(0.4, 0.8), nlpar = "kd", lb = 0)
)

model1<- brm(
    formula = form_main,
    data = all_before_threshold,
    prior = priors,
    chains = 4,
    cores = 4,
    iter = 4000,
    control = list(adapt_delta = 0.95, max_treedepth = 15),
    seed = 123
)



make_clone <- function(df, k) {
  df %>%
    slice(rep(1:n(), each = k)) %>%
    mutate(clone = rep(1:k, times = nrow(df))) %>%
    mutate(
      tree = factor(tree),
      pheno_year = factor(pheno_year),
      tree_year = factor(tree_year),
      clone = factor(clone)
    )
}

# --- function to run a model on cloned data and extract posterior sd & mean
fit_and_summarize <- function(df, k, form, priors, chains = 2, iter = 2000, cores = 2) {
  dclone <- make_clone(df, k)
  message("Fitting k = ", k, " ; nrows = ", nrow(dclone))
  
  fit <- brm(
    formula = form,
    data = dclone,
    prior = priors,
    chains = chains,
    iter = iter,
    cores = cores,
    control = list(adapt_delta = 0.95),
    # you can reduce warmup/iter for quick checks, but increase for final runs
  )

  # Extract posterior summary for parameters of interest:
  ps <- posterior_summary(fit, pars = c("b_kd_Intercept", "b_kf_Intercept",
                                        "b_delta_Intercept", "b_td_Intercept"))
  res <- tibble(
    k = k,
    param = rownames(ps),
    mean = ps[, "Estimate"],
    sd = ps[, "Est.Error"]
  )
  # Return both fit object and summary
  list(fit = fit, summary = res)
}

library(purrr)
ks <- c(1, 5)   # start with these; add larger k if needed
results_pop <- map(ks, ~ fit_and_summarize(data, .x, form_main, priors, chains = 2, iter = 2000, cores = 2))

# combine summaries into a table
summary_pop <- map_dfr(results_pop, "summary")
print(summary_pop)







model<- brm(
    formula = form_main,
    data = data,
    prior = priors,
    chains = 4,
    cores = 4,
    iter = 4000,
    control = list(adapt_delta = 0.95, max_treedepth = 15),
    seed = 123
)

posts <- posterior_samples(model)

post_columns<- colnames(posts)
td_year<- posts[,grepl("r_pheno_year__td", post_columns)]    #there is 7 years here
td_tree<- posts[,grepl("r_tree__td", post_columns)] #there is 15 trees here

colnames(td_tree)<- gsub("r_tree__td\\[(.*),.*", "\\1", colnames(td_tree))
colnames(td_year)<- gsub("r_pheno_year__td\\[(\\d{4}),.*", "\\1", colnames(td_year))



windows()
hist(posts$b_delta_Intercept , breaks=100)


posts$r_tree__td[A,Intercept]
