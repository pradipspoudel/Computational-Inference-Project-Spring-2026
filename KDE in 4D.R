train_500 <- read.csv("datasets/train_500.csv")

library(ks)
fit <- ks::kde(x = train_500)

library(GGally)

# Convert your training data to a dataframe
df_train <- as.data.frame(train_500)
colnames(df_train) <- c("Var1", "Var2", "Var3", "Var4")

ggpairs(df_train,
        upper = list(continuous = wrap("density", alpha = 0.5)),
        lower = list(continuous = wrap("points", alpha = 0.3, size = 0.5)),
        diag = list(continuous = wrap("densityDiag")))

# Create the continuous function
f_hat <- function(x_vec) {
  # x_vec should be a vector of length 4
  predict(fit, x = matrix(x_vec, nrow = 1))
}

# Now you can use it like a math function
#f_hat(c(1, 2, 1, 2))

# Extract the 1D marginal KDE for the first coordinate
# we use the marginal bandwidth from the 4D H matrix
fit_1d <- ks::kde(x = train_500[, 1], h = sqrt(fit$H[1,1]))

# Use the qkde function to find the quantiles directly
lower <- ks::qkde(0.005, fit_1d)
upper <- ks::qkde(0.995, fit_1d)

pi_99_precise <- c(lower, upper)

# -------------------------
# Load test data and compute S1, S2
# -------------------------
test_5000 <- read.csv("datasets/test_5000.csv")

# Make sure test data is numeric matrix with same 4 columns
test_mat <- as.matrix(test_5000)

# S1: average log predictive density on the test set
test_dens <- predict(fit, x = test_mat)
test_dens <- pmax(test_dens, .Machine$double.xmin)  # avoid log(0)
S1 <- mean(log(test_dens))

# S2: interval score for the first coordinate using the 99% interval [lower, upper]
alpha <- 1 - 0.99
y1_test <- test_mat[, 1]

interval_score_each <- (upper - lower) +
  (2 / alpha) * (lower - y1_test) * (y1_test < lower) +
  (2 / alpha) * (y1_test - upper) * (y1_test > upper)

S2 <- -mean(interval_score_each)

# Print results
cat("Score of log prediction density (S1):", S1, "\n")
cat("Interval estimation score (S2):", S2, "\n")
cat("Total score change (S1 + S2):", S1 + S2, "\n")