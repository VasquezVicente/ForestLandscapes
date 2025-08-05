library(pacman)
p_load(dplyr, ggplot2, lubridate, dbscan, lme4, circular, bpnreg,tidyr, tidyverse, bpnreg)


# Convert to DOY
radians_to_days <- function(r) {
  (r / (2 * pi)) * 365
}


set.seed(123)
ids <- LETTERS[1:10]
means <- runif(length(ids), min = 0, max = 2*pi)

# Define 5 clusters, each with its own small circular shift
cluster_shifts <- runif(10, min = -0.1, max = 0.1)  # small shift in radians

# Simulate one observation per individual per cluster
df <- do.call(rbind, lapply(1:10, function(cluster_id) {
  do.call(rbind, lapply(1:length(ids), function(i) {
    mu_total <- circular::circular(means[i] + cluster_shifts[cluster_id])
    theta <- circular::rvonmises(n = 1, mu = mu_total, kappa = 1000)
    data.frame(
      GlobalID = ids[i],
      mean_rad = means[i],
      cluster = cluster_id,
      cluster_shift = cluster_shifts[cluster_id],
      theta_rad = theta
    )
  }))
}))


# Convert to numeric radians for plotting
df$theta_rad <- as.numeric(df$theta_rad)
df$GlobalID_num <- as.numeric(as.factor(df$GlobalID))

View(df)

windows()
ggplot() +
  geom_point(data = df,aes(x = theta_rad, y = 1,color=GlobalID), 
    size = 2, alpha = 0.6
  ) +
  coord_polar(start = 0, direction = 1)


df$cluster<- as.factor(df$cluster)
model <- bpnme(
  pred.I = theta_rad ~ cluster + (1 | GlobalID_num),
  data = df,
  its = 2000,
  burn = 200,
  n.lag = 5
)


results<- as_tibble(model$circ.coef.means)
results<- results[1:5,]
results$cluster<- as.factor(c(1,2,3,4,5))
results$mean<- results$mean %% (2*pi)
View(results)
mean(model$cRI)



all_vals <- as.vector(model$circular.ri)
breaks <- seq(min(all_vals, na.rm = TRUE), max(all_vals, na.rm = TRUE), length.out = 50)
colors <- rainbow(nrow(model$circular.ri))

windows()
hist(model$circular.ri[1, ], breaks = breaks, 
     col = adjustcolor(colors[1], alpha.f = 0.1),
     xlim = c(min(all_vals), max(all_vals)), border = NA,
     main = "Overlapping Histograms of Circular RIs (Dipteryx oleifera)",
     xlab = "Day of Year (converted from radians)", freq = FALSE,
     xaxt = "n")

# Add all remaining histograms with transparency
for (i in 2:nrow(model$circular.ri)) {
  hist(model$circular.ri[i, ], breaks = breaks, 
       col = adjustcolor(colors[i], alpha.f = 0.7),
       border = NA, add = TRUE, freq = FALSE)
}
radian_ticks <- seq(min(all_vals), max(all_vals), length.out = 20)
doy_labels <- round(radians_to_days(radian_ticks %% (2 * pi)))
axis(side = 1, at = radian_ticks, labels = doy_labels)


ri_means <- apply(model$circular.ri, 1, function(row) {
  mean.circular(circular(row))
})

ri_df <- data.frame(
  GlobalID_num = ids,
  theta_mean = ri_means
)

ri_df$theta_mean<- ri_df$theta_mean %% (2*pi)

windows()
ggplot() +
  geom_point(data = df,aes(x = theta_rad, y = 1, color=GlobalID), 
    size = 2, alpha = 0.6
  ) +
  coord_polar(start = 0, direction = 1)+
  geom_segment(data = ri_df,aes(x = theta_mean, y = 0, xend = theta_mean, yend = 1.5, color=GlobalID_num), 
               arrow = arrow(length = unit(0.2, "cm")) , 
               size = 1)+
  geom_segment(
    data = results,
    aes(x = mean, y = 0, xend = mean, yend = 1.75), 
    size = 1
  ) +
geom_point(data = results,
  aes(x = mean, y = 1.75,shape = cluster), 
  size = 3, stroke = 1.2
)+
  scale_x_continuous(limits = c(0, 2 * pi),
    breaks = 2 * pi * (cumsum(c(0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30)) + 15) / 365,
    labels = month.abb
  ) +
  labs(x = NULL, y = NULL,
    title = paste("Mean Day of Year and Mean Resultant Length DUmmy data")
  )


## i need the mean differences
##pull the first 6 column names of beta1
M <- as.matrix(model$beta1)
pheno_years <- colnames(M)[1:5]

# Intercepts
B01 <- model$beta1[, pheno_years[1]]
B02 <- model$beta2[, pheno_years[1]]

# Mode function
posterior_mode <- function(x) {
  d <- density(x)
  d$x[which.max(d$y)]
}

# Empty list to store results
results_list <- list()

# Loop over all unique year pairs
combs <- combn(pheno_years, 2, simplify = FALSE)  # skip intercept

for (i in seq_along(combs)) {
  year_a <- combs[[i]][1]
  year_b <- combs[[i]][2]
  
  # Get coefficients
  a1 <- model$beta1[, year_a]
  a2 <- model$beta2[, year_a]
  
  b1 <- model$beta1[, year_b]
  b2 <- model$beta2[, year_b]
  
  # Compute circular means
  aM <- atan2(B02 + a2, B01 + a1)
  bM <- atan2(B02 + b2, B01 + b1)

  # Posterior difference
  diff_post <- aM - bM

  # circular vairance
  zeta_probe <- sqrt((B01 + b1)^2 + (B02 + b2)^2)^2/4
  var_probe  <- 1 - sqrt((pi * zeta_probe)/2) * exp(-zeta_probe) *
                          (besselI(zeta_probe, 0) + besselI(zeta_probe, 1))
    
  # Plot histograms of aM and bM
  windows()
  hist(bM, breaks = 30, col = rgb(0, 0, 1, 0.5), xlim = c(-pi/3, pi/3),
       main = paste0("Hist: ", year_a, " (aM) vs ", year_b, " (bM)"),
       xlab = "Value")
  hist(aM, breaks = 30, col = rgb(1, 0, 0, 0.5), add = TRUE)
  legend("topright", legend = c(paste0("bM: ", year_b), paste0("aM: ", year_a)),
         fill = c(rgb(0, 0, 1, 0.5), rgb(1, 0, 0, 0.5)))
  
  # Plot histogram of diff_post
  windows()
  hist(diff_post, breaks = 50, col = "gray",
       main = paste0("Posterior of ", year_a, " - ", year_b),
       xlab = paste0("Difference (", year_a, " - ", year_b, ")"))
  abline(v = 0, col = "red", lty = 2)
  mean_diff <- mean(diff_post)
mode_diff <- posterior_mode(diff_post)
sd_diff   <- sd(diff_post)
lb_diff   <- quantile(diff_post, 0.025)
ub_diff   <- quantile(diff_post, 0.975)

# Probabilities relative to direction of effect
if (mean_diff > 0) {
  p_effect_direction <- mean(diff_post > 0)
} else {
  p_effect_direction <- mean(diff_post < 0)
}
p_opposite_direction <- 1 - p_effect_direction

# Save to list
results_list[[i]] <- data.frame(
  YearA = year_a,
  YearB = year_b,
  mean = mean_diff,
  mode = mode_diff,
  sd   = sd_diff,
  LB   = lb_diff,
  UB   = ub_diff,
  DAY  = radians_to_days(mean_diff),
  P_effect_direction = p_effect_direction,
  P_opposite_direction = p_opposite_direction
)
}

# Combine all into one data frame
summary_all_pairs <- do.call(rbind, results_list)

# Print final summary
summary_all_pairs$significance <- ifelse(
  (summary_all_pairs$LB > 0 & summary_all_pairs$UB > 0) |
  (summary_all_pairs$LB < 0 & summary_all_pairs$UB < 0),
  "*",
  ""
)
summary_all_pairs$label_text <- paste0(round(summary_all_pairs$DAY, 0), summary_all_pairs$significance)
View(summary_all_pairs)
windows()
ggplot(summary_all_pairs, aes(x = YearB, y = YearA, fill = DAY, label = label_text)) +
  geom_tile() +
  geom_text(size= 6) +
  scale_fill_gradient2(low = "red2", mid = "white", high = "blue", midpoint = 0,
    name = "Difference in days") +
  scale_x_discrete(
    name = "Comparison Year",
    labels = c("2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024")
  ) +
  scale_y_discrete(
    name = "Reference Year",
    labels = c("2018-2019", "2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024")
  ) +
  theme_minimal()+
  labs(
    title = paste("Difference between all phenological year pairs")
    
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size=10),
    plot.title = element_text(hjust = 0.5, face = "bold", size = 18)
  )
