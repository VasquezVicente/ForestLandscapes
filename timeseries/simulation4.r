library(pacman)
p_load(dplyr, lubridate, ggplot2,brms,tidyr)

f_decay <- function(values, k_d, t_d){
    1/ (1 + exp(k_d *(values - t_d)))
}

f_flush <- function(values, k_f, t_f){
    1/ (1 + exp(-k_f * (values - t_f)))
}

values1 <- seq(0, 100, by=1)

LC <- function(values, k_d, k_f, t_d, t_f) {
  result <- (f_decay(values, k_d, t_d) +
        f_flush(values, k_f, t_f))
  return(result)
}

LC_values<- f_decay(values1, k_d=0.8, t_d=43)

windows()
plot(values1, LC_values, type='l', col='black', ylim=c(0,1))


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
  ) %>%
  filter(pheno_year != "2017")
View(data)

windows()
ggplot(data, aes(x=DOY, y=y_norm, color=tree_year)) +
  geom_line() +
  facet_wrap(~pheno_year,ncol=1)+
  scale_x_continuous(breaks=seq(0,365,30), limits=c(0,365))


trees<- unique(data$tree)
years<- unique(data$pheno_year)

all_before_threshold <- data.frame()
for (i in 1:length(trees)) {
  for (j in 1:length(years)) {
    subset_data <- data %>% filter(tree == trees[i], pheno_year == years[j])
    subset_data<- subset_data %>% arrange(DOY)
    windows()
    plot(subset_data$DOY, subset_data$y_norm, type='l', main=paste("Tree:", trees[i], "Year:", years[j]),
         xlab="Day", ylab="Normalized Leafing")

    #find values below threshold
    threshold <- 0.2
    below_threshold <- which(subset_data$y_norm < threshold)
    last_item<-max(below_threshold)
    #subset all values before last_item
    if (length(below_threshold) > 0 && last_item < nrow(subset_data)) {
      subset_before_threshold <- subset_data[1:last_item, ]
      points(subset_before_threshold$DOY, subset_before_threshold$y_norm, col='red', pch=19)
      all_before_threshold <- bind_rows(all_before_threshold, subset_before_threshold)
    }
}
}

all_before_threshold$pheno_year <- droplevels(all_before_threshold$pheno_year)
levels(all_before_threshold$pheno_year)
windows()
ggplot(all_before_threshold, aes(x=DOY, y=y_norm, color=tree_year)) +
  geom_line()+
  geom_line(data=data.frame(x=values1, y=LC_values), aes(x=x, y=y), color='black', size=1)


class(all_before_threshold$pheno_year)
form_main <- bf(
  y_norm ~ (1 / (1 + exp(kd * (DOY - td)))),
  kd~1,
  td~1 + (1 | pheno_year) + (1 | tree),
  nl = TRUE
)


priors<-c (
    prior(normal(50, 30), nlpar = "td", lb = 0),
    prior(normal(0.6, 1), nlpar = "kd", lb = 0)
)

windows()
hist(rnorm(1000, mean=50, sd=30), breaks=100)


model1<- brm(
    formula = form_main,
    data = all_before_threshold,
    family= Beta(),
    prior = priors,
    chains = 4,
    cores = 4,
    iter = 4000,
    control = list(adapt_delta = 0.95, max_treedepth = 15)
)


View(all_before_threshold)
post <- as_draws_df(model1)
colnames_post<-colnames(post)
#timing parameter td
td_year<- post[,grepl("r_pheno_year__td", colnames_post)]    #there is 7 years here
td_tree<- post[,grepl("r_tree__td", colnames_post)] #there is 15 trees here
td_intercept<- post[,grepl("b_td_Intercept", colnames_post)]


colnames(td_tree)<- gsub("r_tree__td\\[(.*),.*", "\\1", colnames(td_tree))
colnames(td_year)<- gsub("r_pheno_year__td\\[(.*),.*", "\\1", colnames(td_year))
colnames(td_year)[1] <- "2018"

windows()
hist(td_intercept$b_td_Intercept , breaks=100)

year_names <- colnames(td_year)
tree_names <- colnames(td_tree)
td_combinations<- data.frame(rows= 1:nrow(td_year))
for (i in year_names){
    for (j in tree_names){
      td_combinations[,paste(j,i, sep="_")] <- td_intercept$b_td_Intercept + td_year[,i] + td_tree[,j]  ## intercept is td_year[,1]
    }
}

td_combinations<- td_combinations[,-1]

td_long <- td_combinations %>%
  pivot_longer(cols = everything(), names_to = "tree_year", values_to = "td_value") %>%
  separate(tree_year, into = c("tree", "pheno_year"), sep = "_")


windows()
ggplot(td_long, aes(x=td_value, fill=tree)) +
  geom_histogram(aes(y=(..density..)*5), bins=200, alpha=0.5, position="identity") +
  labs(title="Posterior Distributions of td by Pheno Year",
       x="td Value",
       y="Density") +
  geom_line(data= all_before_threshold, aes(x=DOY, y=y_norm,group=tree), color='gray', alpha=0.9) +
  facet_wrap(~pheno_year, ncol=1) +
  theme_minimal()

  scale_x_continuous(breaks=seq(0,200,30), limits=c(0,200))


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
