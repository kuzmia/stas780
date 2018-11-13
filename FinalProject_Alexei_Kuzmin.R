### CLASSIFICATION

# Random Forests
library(randomForest)
library(MASS)
library(tree)
library(gbm)
library(splitstackshape)
library(e1071)
library(mclust)
library(cluster)
library(tidyverse)
library(gridExtra)
library(GGally)

#mt
biomech <- read.csv("column_3C_weka.csv")
biomech$class <- factor(biomech$class)

df <- read.csv("column_3C_weka.csv")
df$class <- factor(df$class)
df$num <- seq(1,310, by = 1)

x<-biomech[,-c(7)]
x <- scale(x)

## descriptive plot

g1 <- ggpairs(biomech, mapping=aes(colour=class))
summary(biomech)

#add PCA
biomech_pca <- prcomp(biomech[,-7], scale = TRUE)
biomech_pca
imp <- summary(biomech_pca)
imp

pairs(biomech_pca$x,col=biomech[,7])
pairs(biomech_pca$x[,1:4],col=biomech[,7])

ari_rf <- c()
ari_mda <- c()
ari_mda_pca <- c()

for(i in 1:10){ #for 10 trials
  set.seed(i)
  y <- stratified(df,"class", 0.75)
  train <- as.vector(y$num)
  
  biomech_rf = tune.randomForest(class~., data = biomech[train,], mtry = 1:6, 
                                 tunecontrol = tune.control(sampling = "cross", cross=5))
  summ <- summary(biomech_rf)
  
  rf.biomech = randomForest(class~.,data=biomech, subset=train, 
                          mtry=summ$best.model$mtry, importance=TRUE, type="class") #continues with best mtry value
  
  biomech.test = biomech[-train,"class"]
  biomech.pred = predict(rf.biomech,biomech[-train,],type="class")
  tab <- table(biomech.test,biomech.pred)
  ari_rf[i] <- classAgreement(tab)$crand
  importance(rf.biomech)
  varImpPlot(rf.biomech, main = "Variable Importance for Random Forests")
  
  ### Mixture Discriminant Analysis
  dfMclustDA <- MclustDA(x[train,], biomech[train,7])
  summary(dfMclustDA, newdata = x[-train,], newclass = biomech[-train,7])
  info <- summary(dfMclustDA, newdata = x[-train,], newclass = biomech[-train,7])
  ari_mda[i] <- classAgreement(info$tab.newdata)$crand

  ### Mixture DA with PCA
  dfMclustDA <- MclustDA(biomech_pca$x[,1:4][train,], biomech[train,7])
  summary(dfMclustDA, newdata = biomech_pca$x[,1:4][-train,], newclass = biomech[-train,7])
  info <- summary(dfMclustDA, newdata = biomech_pca$x[,1:4][-train,], newclass = biomech[-train,7])
  ari_mda_pca[i] <- classAgreement(info$tab.newdata)$crand
}

ari <- data.frame(rbind(ari_rf, ari_mda, ari_mda_pca))
colnames(ari) <- c(seq(1,10, by=1))
rownames(ari) <- c("random forests", "MclustDA", "MclustDA PCA")

for(i in 1:3){
  ari$avg[i] <- rowMeans(ari[i,])
}
ari <- round(ari, digits = 4)

### CLUSTERING

##K-Means/K-Medoids without PCA

biomech_kmeans <- kmeans(x,3)

tbl_means <- table(biomech[,7],biomech_kmeans$cluster)

si2 <- silhouette(biomech_kmeans$cluster, dist(x))
plot(si2, nmax=80, cex.names=0.6, main="Silhouette plot for K-means clusters with PCA")

biomech_kmedoids<-pam(x,3)

tbl_med <- table(biomech[,7],biomech_kmedoids$clustering)

si3 <- silhouette(biomech_kmedoids$clustering, dist(x))
plot(si3, nmax= 80, cex.names=0.6, main="Silhouette plot for k-medoids clusters with PCA")

par(mfrow=c(1,2))# Have both silhouettes in one plot
plot(si2, nmax= 80, cex.names=0.6, main="Silhouette plot for K-means clusters")
plot(si3, nmax= 80, cex.names=0.6, main="Silhouette plot for k-medoids clusters")

# Produce an elbow plot
par(mfrow=c(1,1))
K<-8
wss<-rep(0,K)
for (k in 1:K){
  wss[k] <- sum(kmeans(x,k)$withinss)
}
plot(1:K, wss, typ="b", ylab="Total within cluster sum of squares", xlab="Number of clusters (k)")

##K-Means/K-Medoids with PCA

#add PCA
biomech_pca <- prcomp(biomech[,-7], scale = TRUE)
biomech_pca
summary(biomech_pca)

pairs(biomech_pca$x,col=biomech[,7])
pairs(biomech_pca$x[,1:4],col=biomech[,7])

# kmeans
biomech_kmeans<-kmeans(biomech_pca$x[,1:4],3)
biomech_kmeans
biomech_kmeans$cluster

tbl_means <- table(biomech[,7],biomech_kmeans$cluster)
tbl_means

si2 <- silhouette(biomech_kmeans$cluster, dist(biomech_pca$x[,1:4]))
plot(si2, nmax=80, cex.names=0.6, main="Silhouette plot for K-means clusters with PCA")

biomech_kmedoids<-pam(biomech_pca$x[,1:4],3)
biomech_kmedoids
biomech_kmedoids$clustering

tbl_med <- table(biomech[,7],biomech_kmedoids$clustering)
tbl_med

si3 <- silhouette(biomech_kmedoids$clustering, dist(biomech_pca$x[,1:4]))
plot(si3, nmax= 80, cex.names=0.6, main="Silhouette plot for k-medoids clusters with PCA")

par(mfrow=c(1,2)) # Have both silhouette plots in one plot
plot(si2, nmax= 80, cex.names=0.6, main="Silhouette plot for K-means clusters with PCA")
plot(si3, nmax= 80, cex.names=0.6, main="Silhouette plot for k-medoids clusters with PCA")


### ROC analysis

library(pgmm)
library(MixGHD)
library(tidyverse)
library(ROCR)

biomech <- read.csv("column_2C_weka.csv")
biomech$class <- factor(biomech$class)
biomech$cls <- rep(0,nrow(biomech))

for(i in 1:310){
  if(biomech$class[i]=="Abnormal"){biomech$cls[i]<-1}
  else if(biomech$class[i]=="Normal"){biomech$cls[i]<-0}
}

x<-biomech[,-c(7,8)]
x <- scale(x)
cls <-biomech$cls+1
for(i in 1:310){
  if(i%%3==0){cls[i]<-0}
}

biomech_class <- pgmmEM(x,2:2,1:2,cls,relax=TRUE)
cls_ind <- (cls==0) 

tab <- table(biomech[cls_ind,7],biomech_class$map[cls_ind])
tab2 <- tab[,c(2,1)]


biomech_class$zhat[cls_ind,1]

cls2<-(biomech[cls_ind,8]+1)%%2
pred<-prediction(biomech_class$zhat[cls_ind,1],cls2)
perf <- performance(pred, "tpr", "fpr")
plot(perf,ylab="Sensitivity", xlab="1-Specificity")
abline(0,1, lty=3, lwd=2)
text(x=0.89, y= 0, labels = "AUC: ")
perf <- performance(pred, measure = "auc")
text(x=0.94, y=0, labels = round(as.numeric(perf@y.values), digits = 4))

# above we create a predictive model using pgmm because of its embedded factor analysis.
# diagnostic value is implied because visually we can see that the AUC > 0.5.
# AUC > 0.5 means that our model classifies better than chance.
