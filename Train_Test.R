
install.packages("e1071")
library(e1071)
install.packages("randomForest")
library(randomForest)

### CV RF 
data.training <- read.csv("data.trainning_new_20260611.csv")
library(randomForest)
set.seed(11234)

vars <- c("Ni","Cr","Fe","Co","Al","Ti","Mo","V","W","Ta","Nb",
          "homogenize_K","CR_01","recrystalize_K","CR_02","annealing_K",
          "aging_K","ag.time_h","HR","omiga","VEC","delta_H","IED","PR")

n <- nrow(data.training)
folds <- sample(rep(1:5, length.out = n))
pred.cv <- rep(NA, n)

# Hyperparameter grid to search
mtry_grid    <- c(4, 6, 8, 10, 12, ceiling(length(vars)/3))
nodesize_grid <- c(1, 3, 5)
ntree_val    <- 1000  # set high; OOB error stabilizes, no need to tune this separately

for (k in 1:5) {
  train.k <- data.training[folds != k, ]
  test.k  <- data.training[folds == k, ]
  
  # --- Inner tuning using OOB error (no leakage, no extra CV loop needed) ---
  best_oob <- Inf
  best_mtry <- NA
  best_nodesize <- NA
  
  for (m in mtry_grid) {
    for (ns in nodesize_grid) {
      rf_tmp <- randomForest(
        x = train.k[, vars],
        y = train.k$ys_Mpa,
        mtry = m,
        nodesize = ns,
        ntree = ntree_val
      )
      oob_mse <- tail(rf_tmp$mse, 1)  # OOB MSE at final tree
      if (oob_mse < best_oob) {
        best_oob <- oob_mse
        best_mtry <- m
        best_nodesize <- ns
      }
    }
  }
  
  cat("Fold", k, "- best mtry:", best_mtry, " best nodesize:", best_nodesize, "\n")
  
  # --- Fit final RF for this fold using best hyperparameters ---
  rf.k <- randomForest(
    x = train.k[, vars],
    y = train.k$ys_Mpa,
    mtry = best_mtry,
    nodesize = best_nodesize,
    ntree = ntree_val
  )
  
  pred.cv[folds == k] <- predict(rf.k, test.k[, vars])
}

plot(data.training$ys_Mpa, pred.cv,
     xlab = "Observed ys_Mpa", ylab = "CV Predicted ys_Mpa",
     main = "5-Fold CV (Tuned RF)")
abline(0, 1, col = "red", lty = 2)

rmserf.cv <- sqrt(mean((data.training$ys_Mpa - pred.cv)^2))
rmserf.cv
### XG Boost of Aug 2026
data.training <-read.csv("data.trainning_new_20260611.csv")
library(xgboost)
set.seed(11234)
vars <- c("Ni","Cr","Fe","Co","Al","Ti","Mo","V","W","Ta","Nb","homogenize_K","CR_01","recrystalize_K","CR_02","annealing_K","aging_K","ag.time_h","HR","omiga","VEC","delta_H","IED","PR")

n<-nrow(data.training)
folds<-sample(rep(1:4,length.out=n))

pred.cv<-rep(NA,n)

for(k in 1:4){
  train.k<-data.training[folds!=k,]
  test.k<-data.training[folds==k,]
  
  xgb.k<-xgboost(data=as.matrix(train.k[,vars]), label=train.k$ys_Mpa, nrounds=500, max_depth=6, eta=0.05, objective="reg:squarederror", verbose=0)
  
  pred.cv[folds==k]<-predict(xgb.k,as.matrix(test.k[,vars]))
}

plot(data.training$ys_Mpa, pred.cv)
abline(0,1)
rmsexgb.cv<-sqrt(sum((data.training$ys_Mpa-pred.cv)^2)/n)
rmsexgb.cv

### CV of ANN
data.training <-read.csv("data.trainning_new_20260611.csv")
#
library(nnet)
set.seed(11234)
vars <- c("Ni","Cr","Fe","Co","Al","Ti","Mo","V","W","Ta","Nb","homogenize_K","CR_01","recrystalize_K","CR_02","annealing_K","aging_K","ag.time_h","HR","omiga","VEC","delta_H","IED","PR")

n<-nrow(data.training)
folds<-sample(rep(1:4,length.out=n))

pred.cv<-rep(NA,n)

for(k in 1:4){
  train.k<-data.training[folds!=k,]
  test.k<-data.training[folds==k,]
  
  ann.k<-nnet(train.k$ys_Mpa ~ ., data=train.k[,vars], size=10, linout=TRUE, maxit=1000, decay=0.01, trace=FALSE)
  
  pred.cv[folds==k]<-predict(ann.k,test.k[,vars])
}

plot(data.training$ys_Mpa, pred.cv)
abline(0,1)
rmseann.cv<-sqrt(sum((data.training$ys_Mpa-pred.cv)^2)/n)
rmseann.cv


### SVR.rbf 

data.training <- read.csv("data.trainning_new_20260611.csv")
library(e1071)
set.seed(123456)

f <- ys_Mpa ~ Ni+Cr+Fe+Co+Al+Ti+Mo+V+W+Ta+Nb+homogenize_K+CR_01+recrystalize_K+
  CR_02+annealing_K+aging_K+ag.time_h+HR+omiga+VEC+delta_H+IED+PR

tuned_parameters <- tune.svm(f, data = data.training,
                             kernel = "radial",
                             gamma = 10^(-5:5), cost = 10^(-3:3))
summary(tuned_parameters)

best.gamma <- tuned_parameters$best.parameters$gamma
best.cost  <- tuned_parameters$best.parameters$cost

n <- nrow(data.training)
folds <- sample(rep(1:5, length.out = n))

pred.cv <- rep(NA, n)

for (k in 1:5) {
  train.k <- data.training[folds != k, ]
  test.k  <- data.training[folds == k, ]
  
  svr.k <- svm(f, data = train.k,
               kernel = "radial",          # was "linear": this was the bug
               cost = best.cost, gamma = best.gamma)
  
  pred.cv[folds == k] <- predict(svr.k, test.k)
}

plot(data.training$ys_Mpa, pred.cv)
abline(0, 1)
rmsesvr.cv <- sqrt(mean((data.training$ys_Mpa - pred.cv)^2))
rmsesvr.cv


#### SVR. linear
data.training <-read.csv("data.trainning_new_20260611.csv")
library(e1071)
set.seed(123456)

n<-nrow(data.training)
folds<-sample(rep(1:5,length.out=n))

pred.cv<-rep(NA,n)

for(k in 1:5){
  train.k<-data.training[folds!=k,]
  test.k<-data.training[folds==k,]
  
  svr.k<-svm(ys_Mpa ~Ni+Cr+Fe+Co+Al+Ti+Mo+V+W+Ta+Nb+homogenize_K+CR_01+recrystalize_K+CR_02+annealing_K+aging_K+ag.time_h+HR+omiga+VEC+delta_H+IED+PR,
             data=train.k,kernel="linear")
  #,cost=best.cost,gamma=best.gamma)
  
  pred.cv[folds==k]<-predict(svr.k,test.k)
}

plot(data.training$ys_Mpa, pred.cv)
abline(0,1)
rmsesvr.cv<-sqrt(sum((data.training$ys_Mpa-pred.cv)^2)/n)
rmsesvr.cv


#### XGBoost CV
data.training <- read.csv("data.trainning_new_20260611.csv")
library(xgboost)
set.seed(11234)

vars <- c("Ni","Cr","Fe","Co","Al","Ti","Mo","V","W","Ta","Nb",
          "homogenize_K","CR_01","recrystalize_K","CR_02","annealing_K",
          "aging_K","ag.time_h","HR","omiga","VEC","delta_H","IED","PR")

X <- as.matrix(data.training[, vars])
y <- data.training$ys_Mpa

# Create 5 folds
k <- 5
folds <- sample(rep(1:k, length.out = nrow(X)))

# Storage for out-of-fold predictions
cv_pred <- numeric(nrow(X))

for (i in 1:k) {
  train_idx <- which(folds != i)
  test_idx  <- which(folds == i)
  
  xgb_cv <- xgboost(
    data = X[train_idx, ],
    label = y[train_idx],
    nrounds = 500,
    max_depth = 6,
    eta = 0.05,
    objective = "reg:squarederror",
    verbose = 0
  )
  
  cv_pred[test_idx] <- predict(xgb_cv, X[test_idx, ])
}

# Plot observed vs. cross-validated predicted
plot(y, cv_pred,
     xlab = "Observed ys_Mpa",
     ylab = "CV Predicted ys_Mpa",
     main = "5-Fold Cross-Validation")
abline(0, 1, col = "red", lty = 2)

# RMSE from cross-validation
rmsexgb_cv <- sqrt(mean((y - cv_pred)^2))
rmsexgb_cv
