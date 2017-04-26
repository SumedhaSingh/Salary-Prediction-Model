#Reading the training data
dtm_features <- read.csv('train_features_2013-03-07.csv', header = T)
dtm_salaries <- read.csv('train_salaries_2013-03-07.csv', header = T)

dtm <- merge(dtm_features, dtm_salaries, by ="jobId")

#Converting the categories into numeric factors
dtm$jobTypeFact <- as.numeric(factor(dtm$jobType , levels=c("CEO","CFO","CTO","JANITOR","JUNIOR","MANAGER","SENIOR","VICE_PRESIDENT")))
dtm$degreeFact <- as.numeric(factor(dtm$degree , levels=c("BACHELORS","DOCTORAL","HIGH_SCHOOL","MASTERS","NONE")))
dtm$industryFact <- as.numeric(factor(dtm$industry , levels=c("AUTO","EDUCATION","FINANCE","HEALTH","OIL","SERVICE","WEB")))
dtm$majorFact <- as.numeric(factor(dtm$major , levels=c("BIOLOGY","BUSINESS","CHEMISTRY","COMPSCI","ENGINEERING","LITERATURE","MATH","NONE","PHYSICS")))

#Divide the data into training and testing into 70/30
set.seed(14)
random= sample(2,nrow(dtm),replace=T,prob = c(0.70,0.30))
train = dtm[random==1,]
test= dtm[random==2,]

#Features Selected
varNames <- names(train)
features <- varNames[varNames %in% c("yearsExperience", "milesFromMetropolis", "jobTypeFact", "degreeFact","industryFact", "salary")]
dataWithoutPredicted = train[features]


#random forest model building
install.packages("randomForest")
library(randomForest)

rf <- randomForest(salary~.,data=dataWithoutPredicted, ntree=100, mtry=2, keep.forest=TRUE ,importance=TRUE, replace=FALSE)

#Variable importance
varImpPlot(rf)
plot(rf)
summary(rf)

#Test with selective features
testWithoutPredicted = test[features]

#Testing model with the test features 
randomPredict <- predict(rf, newdata = testWithoutPredicted)
randomPredict

#Writing the result salaries
write.csv(randomPredict, file = "test_salaries.csv")

#Linear Regression Model
regModel=lm(train$salary~train$milesFromMetropolis+train$yearsExperience+train$industry+train$degree+train$industry,data=train)
summary(regModel)

#Decision tree
install.packages("rpart")
library(rpart)
fit <- rpart(train$salary~train$milesFromMetropolis+train$yearsExperience+train$industry+train$industry,data=train)
printcp(fit) # display the results 
plotcp(fit) # visualize cross-validation results 
summary(fit) # detailed summary of splits

dtree=predict(fit,newdata = test, interval = "confidence", type = 'class')
write.csv(dtree, file = "dtree_ouput.csv")
