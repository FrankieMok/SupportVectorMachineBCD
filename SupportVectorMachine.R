# Evulate the Breast-Cancer-Wisconsin-Diagnostic-DataSet

# set my working path
#setwd("D:/PGD_Computing/ISCG8050/R_prog")

# read Dataset and add column names
#install.packages("data.table")
library("data.table")
datasets <- fread('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data')
#datasets <- read.csv("wdbc.data", header = F)
colnames(datasets) <- c("id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
                        "compactness_mean","concavity_mean","concave_points_mean","symmetry_mean", "fractal_dimension_mean",
                        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se",
                        "concave_points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst",
                        "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave_points_worst",
                        "symmetry_worst", "fractal_dimension_worst")
datasets$diagnosis <- factor(datasets$diagnosis)

##----------------------------------------------------------------------------------------------------------

# Check and observe the datasets
head(datasets)
tail(datasets)
str(datasets)

# check if datasets has NULL, NA or duplicated data
sum(is.null(datasets))
sum(is.na(datasets))
datasets[duplicated(datasets)]

# Exploratory data analysis
sum(datasets$diagnosis=="B") # Begnign =357 so Malign = 212
## gglot2 https://www.youtube.com/watch?v=rsG-GgR0aEY
library(ggplot2)

##------ Visual analysis ------- Create Plots ----------------------------

# hist plot
ggplot(datasets) + aes(datasets$diagnosis) + geom_bar(color="black", fill="lightblue", linetype="dashed") + ggtitle("Breast Cancer Diagnosis") +
  xlab("Benign & Malignant") + ylab("Count") + geom_text(stat='count', aes(label=..count..), vjust=4) + theme(
    plot.title = element_text(color="Black", size=20, face="bold.italic"),
    axis.title.x = element_text(color="blue", size=14, face="bold"),
    axis.title.y = element_text(color="#800000", size=14, face="bold")
  )
#pdf('myplot.pdf')
pairs(datasets[,3:12], main = "Breast Cancer  Data -- 10 parameters", pch = 21, bg = c("red", "green3")[unclass(datasets$diagnosis)])
#dev.off()
plot(datasets$radius_mean,pch = 21,bg = c("red", "green3")[unclass(datasets$diagnosis)])
install.packages("reshape2")
library(reshape2)
long <- melt(datasets[,-1])
boxplot(value ~ variable, data=long, horizontal=TRUE, main = "Breast Cancer Boxpot -- 30 features")

# install Normalizes package
#install.packages("BBmisc")
library(BBmisc)
datasets <- normalize(datasets, method = 'standardize', range = c(0,1))

long.B <- melt(subset(datasets, diagnosis=="B")[,3:12])
boxplot(value ~ variable, data=long.B, main = "Breast Cancer Boxpot -- Benign", col="blue")
long.M <- melt(subset(datasets, diagnosis=="M")[,3:12])
boxplot(value ~ variable, data=long.M, main = "Breast Cancer Boxpot -- Malignant", col = "red")

#install.packages("corrplot")
#install.packages("Hmisc")
library(corrplot) # corrplot function to create plot
library(Hmisc) # rcorr function
datasets.matrix <- data.matrix(datasets[,3:12])
datasets.cor = cor(datasets.matrix , method = c("spearman"))
datasets.rcorr = rcorr(as.matrix(datasets.cor))
corrplot(datasets.rcorr$r)

## --------- Separate the Datasets in Training, Cross Valiation, Testing -------

# Start SVM simulation
set.seed(1102)
ss <- sample(1:3,size=nrow(datasets),replace=TRUE,prob=c(0.6,0.15,0.25))
datasets.train <- datasets[ss==1,]
datasets.cvr <- datasets[ss==2,]
datasets.test <- datasets[ss==3,]

##------------------------------------------------------------------------------------------------------

##  Start the brief Machine Learning analysis--------------------------

#install.packages("caret")
#install.packages("e1071")
library(caret)
library(e1071)

## Linear Kernel
svm.model.lin.t <- svm(diagnosis~., data = datasets.train[,-1], kernel = "linear") # linear
confusionMatrix(datasets.train$diagnosis, predict(svm.model.lin.t))
svm.model.lin.plot.t <- svm(diagnosis~ smoothness_mean + concavity_mean, data = datasets.train[,-1], kernel = "linear") # linear
plot(svm.model.lin.plot.t, datasets.train, smoothness_mean ~ concavity_mean,
     color.palette = topo.colors)
pred.svm.model.lin.t <- predict(svm.model.lin.t, datasets.test[,c(-1,-2)])
confusionMatrix(datasets.test$diagnosis, pred.svm.model.lin.t)

## Radial Kernel
svm.model.rad.t <- svm(diagnosis~., data = datasets.train[,-1], kernel = "radial")
confusionMatrix(datasets.train$diagnosis, predict(svm.model.rad.t))
svm.model.rad.plot.t <- svm(diagnosis~ smoothness_mean + concavity_mean, data = datasets.train[,-1], kernel = "radial")
plot(svm.model.rad.plot.t, datasets.train, smoothness_mean ~ concavity_mean,
     color.palette = topo.colors)
pred.svm.model.rad.t <- predict(svm.model.rad.t, datasets.test[,c(-1,-2)])
confusionMatrix(datasets.test$diagnosis, pred.svm.model.rad.t)

## polynomial Kernel
svm.model.pol.t <- svm(diagnosis~., data = datasets.train[,-1], kernel = "polynomial", degree=1)
confusionMatrix(datasets.train$diagnosis, predict(svm.model.pol.t))
svm.model.pol.plot.t <- svm(diagnosis~ smoothness_mean + concavity_mean, data = datasets.train[,-1], kernel = "polynomial")
plot(svm.model.pol.plot.t, datasets.train, smoothness_mean ~ concavity_mean,
     color.palette = topo.colors)
pred.svm.model.pol.t <- predict(svm.model.pol.t, datasets.test[,c(-1,-2)])
confusionMatrix(datasets.test$diagnosis, pred.svm.model.pol.t)


## sigmoid Kernel
svm.model.sig.t <- svm(diagnosis~., data = datasets.train[,-1], kernel = "sigmoid")
confusionMatrix(datasets.train$diagnosis, predict(svm.model.sig.t))
svm.model.sig.plot.t <- svm(diagnosis~ smoothness_mean + concavity_mean, data = datasets.train[,-1], kernel = "sigmoid")
plot(svm.model.sig.plot.t, datasets.train, smoothness_mean ~ concavity_mean,
     color.palette = topo.colors)
pred.svm.model.sig.t <- predict(svm.model.sig.t, datasets.test[,c(-1,-2)])
confusionMatrix(datasets.test$diagnosis, pred.svm.model.sig.t)

## Cross validation -----------------------------

# Create svm model
svm.model.rad <- svm(diagnosis~., data = datasets.train[,-1], kernel = "radial")
confusionMatrix(datasets.train$diagnosis, predict(svm.model.rad))
pred.svm.rad.test <- predict(svm.model.rad, datasets.test[,c(-1,-2)])
confusionMatrix(datasets.test$diagnosis, pred.svm.rad.test)


svm.tune.rad.cost <- tune(svm, diagnosis ~., data = datasets.cvr[,-1], kernel="radial",
                          ranges = list(cost = seq(0.1, 1, length=30)))
plot(svm.tune.rad.cost)
svm.tune.rad.cost

svm.tune.rad.gamma <- tune(svm, diagnosis ~., data = datasets.cvr[,-1], kernel="radial",
                           ranges = list(gamma = seq(0.01, 0.1, length=20)))
plot(svm.tune.rad.gamma)
svm.tune.rad.gamma

svm.tune.rad.epsilon <- tune(svm, diagnosis ~., data = datasets.cvr[,-1], kernel="radial",
                             ranges = list(epsilon = seq(0.0001, 0.1, length=10)))
plot(svm.tune.rad.epsilon)
svm.tune.rad.epsilon

svm.tune.rad <- tune(svm, diagnosis ~., data = datasets.cvr[,-1], kernel="radial",
                     ranges = list(gamma = seq(0.01, 0.03, length=10),
                                   cost=seq(0.8,1.2,length=40),
                                   epsilon =seq(0.005,0.02,length=10)))
svm.tune.rad

svm.model.rad.tune <- svm(diagnosis~., data = datasets.train[,-1], kernel = "radial", cost = 0.8, gamma = 0.01667, epsilon = 0.005)

pred.svm.rad.tune.test <- predict(svm.model.rad.tune, datasets.test[,c(-1,-2)])
confusionMatrix(datasets.test$diagnosis, pred.svm.rad.tune.test)

### Bias vs Variance
cva <- NULL
cvalue <- c(0.025, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300)
for (i in 1:10) {
  svm.model.rad.tune.bv <- svm(diagnosis~., data = datasets.train[,-1], kernel = "radial", cost = cvalue[i], gamma = 0.01667, epsilon = 0.005)
  cmt <- confusionMatrix(datasets.train$diagnosis, predict(svm.model.rad.tune.bv))
  cva <- c(cva,cmt$overall[[1]])
}

cvb <- NULL
for (i in 1:10) {
  svm.model.rad.tune.bv <- svm(diagnosis~., data = datasets.train[,-1], kernel = "radial", cost = cvalue[i], gamma = 0.01667, epsilon = 0.005)
  pred.svm.rad.tune.bv <- predict(svm.model.rad.tune.bv, datasets.test[,c(-1,-2)])
  cm <- confusionMatrix(datasets.test$diagnosis, pred.svm.rad.tune.bv)
  cvb <- c(cvb,cm$overall[[1]])
}

# Plot the bias vs variance
plot(cva,type = "o", xaxt='n',yaxt='n',ann=FALSE)
par(new=TRUE)
plot(cvb,type = "o",col="red", xaxt='n', xlab = "cost", ylab = "accuracy", main = "Bias vs. Variance")
legend("bottomright", pch=1, col=c("black", "red"), legend = c("Training accuracy", "Testing accuracy"))
axis(1, 1:10, c("0.025","0.03","0.1","0.3","1","3","10","30","100","300"), col.axis = "blue")


## plot ROC curve - Receiver Operating Characteristic (ROC)
#install.packages("pROC")
library(pROC)
pred.svm.rad.test <- predict(svm.model.rad, datasets.test[,c(-1,-2)], decision.values=TRUE)
dv <- attributes(pred.svm.rad.test)$decision.values
plot.roc((as.numeric(datasets.test$diagnosis)-1), dv, xlab = "1 - Specificity", print.auc=TRUE, col="Magenta")
#par(new=TRUE)
pred.svm.rad.tune.test <- predict(svm.model.rad.tune, datasets.test[,c(-1,-2)], decision.values=TRUE)
dv.tune <- attributes(pred.svm.rad.tune.test)$decision.values
plot.roc((as.numeric(datasets.test$diagnosis)-1), dv.tune, xlab = "1 - Specificity", print.auc=TRUE, col="blue",print.auc.y = .4, add = TRUE)


## decision Tree
#install.packages("rpart")
#install.packages("rpart.plot")
library(rpart)
library(rpart.plot)
dt.model<- rpart(diagnosis~., data = datasets.train[,-1])
prp(dt.model, faclen=0, fallen.leaves = T, shadow.col = "gray", extra=2)
pred.dt <- predict(dt.model, newdata=datasets.test[,c(-1,-2)], type="class", decision.value=TRUE)
table(real=datasets.test$diagnosis, predict=pred.dt)
confusionMatrix(datasets.test$diagnosis, pred.dt)

## Random Forest
#install.packages("randomForest")
library(randomForest)
rf.model<- randomForest(datasets.train$diagnosis ~., data = datasets.train[,-1],importance = TRUE)
pred.rf <- predict(rf.model, newdata=datasets.test[,c(-1,-2)], type = "class")
confusionMatrix(datasets.test$diagnosis, pred.rf)

#pred.dt.roc <- predict(dt.model, datasets.test[,c(-1,-2)], type="prob", decision.value=TRUE)
#auc.dt <- auc((as.numeric(pred.dt)-1), pred.dt.roc[,1])
#plot.roc((as.numeric(pred.dt)-1), pred.dt.roc[,1], xlab = "1 - Specificity", print.auc=TRUE, col="red")

