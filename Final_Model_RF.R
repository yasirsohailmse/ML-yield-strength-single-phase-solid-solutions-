install.packages("tidyverse")
library(tidyverse)
#install.packages("ggplot2")
library(ggplot2)
install.packages("corrplot")
library(corrplot)
install.packages("minerva")
library(minerva)
install.packages("GGally")
library(GGally)
source("https://raw.githubusercontent.com/briatte/ggcorr/master/ggcorr.R")
#The main package dependency of ggcorr is the ggplot2 package for plot construction.
#install.packages("dplyr")
library(dplyr)
install.packages("predvars")
library(predvars)
install.packages("e1071")
library(e1071) 
library(readr)
library(dplyr)
install.packages("randomForest")
library(randomForest)
fn.EI =  function(data.training, data.virtual){
  ego = (data.virtual[,"mean"] - max(data.training[,"ys_Mpa"]))/data.virtual[,"sd"]
  z = ego
  ei.ego = data.virtual[,"sd"]*z*pnorm(z) + data.virtual[,"sd"]*dnorm(z) 
  kg = -1*abs((data.virtual[,"mean"] - max(max(data.virtual[,"mean"],max(data.training[,"ys_Mpa"]))))/data.virtual[,"sd"])
  z = kg
  ei.kg = data.virtual[,"sd"]*z*pnorm(z) + data.virtual[,"sd"]*dnorm(z)  
  
  max.P = pnorm(ego, mean = data.virtual[,"mean"], sd = data.virtual[,"sd"])
  
  ei = cbind(data.virtual, ei.ego, ei.kg, max.P)
  ei = data.frame(ei)
  return (ei)
}

#virtual data
data.virtual<- read.csv("data.virual.csv")
# Training data
data.training <-read.csv("data.csv")

# Random forest

set.seed(11234)

rf<-randomForest(ys_Mpa ~ Ni+Cr+Fe+Co+Al+Ti+Mo+V+W+Ta+Nb+homogenize_K+CR_01+recrystalize_K+CR_02+annealing_K+aging_K+ag.time_h+HR+omiga+VEC+delta_H+IED+PR,
                 data=data.training)
plot(data.training$ys_Mpa, predict(rf,data.training))
rmserf<-sqrt(sum((data.training$ys_Mpa-predict(rf,data.training))^2)/length(data.training$ys_Mpa))
#abline(0,1)

#Prediction
predict.data<-predict(rf,data.virtual)
#write.csv(predict.data, file="data.training.predict_01.csv")

R=1000
i=1
repeat{
  #    boot.sample = c(sample(n, 5, replace = FALSE), sample(n, n-5, replace = TRUE))
  set.seed(i)
  boot.sample = sample(329, 329, replace = TRUE)
  t.data.training = data.training[boot.sample,]
  rf<-randomForest(ys_Mpa ~ Ni+Cr+Fe+Co+Al+Ti+Mo+V+W+Ta+Nb+homogenize_K+CR_01+recrystalize_K+CR_02+annealing_K+aging_K+ag.time_h+HR+omiga+VEC+delta_H+IED+PR,
                   data=t.data.training )
  predict.data1 = predict(rf,data.virtual)
  predict.data = cbind(predict.data, predict.data1)
  i=i+1
  if(i>R) break ()}
#mean
mean <-apply(as.matrix(predict.data),1,mean)
data.virtual$mean <- mean
head(mean)
#sd
sd <-apply(as.matrix(predict.data),1,sd)
data.virtual$sd <- sd
head(sd)

output2 <- fn.EI(data.training, data.virtual)
write.csv(output2, file="data.virual_pred.csv")



