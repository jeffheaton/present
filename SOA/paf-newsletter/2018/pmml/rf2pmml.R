# Libraries - https://henning.kropponline.de/2015/09/06/jpmml-example-random-forest/
library(randomForest)
library(XML)
library(pmml)

# data to build model on
data(iris)

# train a model on a 75-25 split between training and validation
z <- sample(2,nrow(iris),replace=TRUE,prob=c(0.75,0.25))
trainData <- iris[z==1,]
testData <- iris[z==2,]

# train model
rf <- randomForest(Species~.,data=trainData,
                   ntree=100,proximity=TRUE)
table(predict(rf),trainData$Species)


# convertto pmml
pmml <- pmml(iris_rf,name="Iris Random Forest",data=iris_rf)

# save PMML XML
saveXML(iris_rf.pmml,"iris.pmml")
