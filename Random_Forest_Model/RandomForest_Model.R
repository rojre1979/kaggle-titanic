# Robert J. Regalado - 10 Dec 2016
# Titanic: Random Forests


# Set working directory and import datafiles
setwd("C:/Users/hp/Desktop/Kaggle/titanic-master")
train <- read.csv("train.csv")
test <- read.csv("test.csv")

# Install and load required packages for decision trees and forests

library(rpart)
install.packages('randomForest')
library(randomForest)
install.packages('party')
library(party)

# Join together the test and train sets for easier feature engineering
test$Survived <- NA
c <- rbind(train, test)

# Convert to a string
c$Name <- as.character(c$Name)


# Engineered variable: Title
c$Title <- sapply(c$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
c$Title <- sub(' ', '', c$Title)


# Combine small title groups
c$Title[c$Title %in% c('Mme', 'Mlle')] <- 'Mlle'
c$Title[c$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
c$Title[c$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'

# Convert to a factor
c$Title <- factor(c$Title)

# Engineered variable: Family size
c$FamilySize <- c$SibSp + c$Parch + 1

# Engineered variable: Family
c$Surname <- sapply(c$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
c$FamilyID <- paste(as.character(c$FamilySize), c$Surname, sep="")
c$FamilyID[c$FamilySize <= 2] <- 'Small'

# Delete erroneous family IDs
famIDs <- data.frame(table(c$FamilyID))
famIDs <- famIDs[famIDs$Freq <= 2,]
c$FamilyID[c$FamilyID %in% famIDs$Var1] <- 'Small'

# Convert to a factor
c$FamilyID <- factor(c$FamilyID)

# Fill in Age NAs
summary(c$Age)
Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize, 
                data=c[!is.na(c$Age),], method="anova")
c$Age[is.na(c$Age)] <- predict(Agefit, c[is.na(c$Age),])
# Check what else might be missing
summary(c)

# Fill in Embarked blanks
summary(c$Embarked)
which(c$Embarked == '')
c$Embarked[c(62,830)] = "S"
c$Embarked <- factor(c$Embarked)

# Fill in Fare NAs
summary(c$Fare)
which(is.na(c$Fare))
c$Fare[1044] <- median(c$Fare, na.rm=TRUE)

# New factor for Random Forests, only allowed <32 levels, so reduce number
c$FamilyID2 <- c$FamilyID

# Convert back to string
c$FamilyID2 <- as.character(c$FamilyID2)
c$FamilyID2[c$FamilySize <= 3] <- 'Small'

# And convert back to factor
c$FamilyID2 <- factor(c$FamilyID2)

# Split back into test and train sets
train <- c[1:891,]
test <- c[892:1309,]

# Build Random Forest Ensemble
set.seed(415)
fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID2,
                    data=train, importance=TRUE, ntree=2000)
# Look at variable importance
varImpPlot(fit)

# Now let's make a prediction and write a submission file
Prediction <- predict(fit, test)
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "firstforest.csv", row.names = FALSE)

# Build condition inference tree Random Forest
set.seed(415)
fit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID,
               data = train, controls=cforest_unbiased(ntree=2000, mtry=3)) 
# Now let's make a prediction and write a submission file
Prediction <- predict(fit, test, OOB=TRUE, type = "response")
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "my_submission_rf.csv", row.names = FALSE)

