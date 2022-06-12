install.packages("readxl")
library("readxl")
library(rsample)
library(recipes)
library(tidyr)
library(stringr)
library(readr)
library(dplyr)
library(mfx)
library(DescTools)
library(LogisticDx)
library(blorr)
library(ggplot2)
library("sandwich")
library("lmtest")
library("MASS")
library("mfx")
library("BaylorEdPsych")
library("htmltools")
library("LogisticDx")
library("aod")
library("logistf")

options(scipen = 999)
setwd("C:/Users/tetiana.heorhiichuk/Desktop")
# Data Exploration --------------------------------------------------------


# loading data
data <- read.csv("Churn_Modelling.csv")
View(data)

# dimension of the data. 
# our dataset consist 10000 rows and 14 columns.
dim(data)
head(data)
colnames(data)


# checking for missing values. 
# our data does not consist any missing value 
colSums(is.na(data))

# counts for different exited
data %>% count(data$Exited)

# minimum and maximum age
min(data$Age)
max(data$Age)

# Explorary data analysis --------------------------------------------------------

data %>% 
  mutate(Exited = str_replace(Exited, '\\.', '')) %>%
           ggplot(aes(x = Age, fill = Exited)) + geom_histogram() + theme_bw()

data %>% 
  mutate(Exited = str_replace(Exited, '\\.', '')) %>%
  ggplot(aes(x = Gender, fill = Exited)) + geom_bar(position="dodge") + theme_bw() + coord_flip()

data %>% 
  mutate(Exited = str_replace(Exited, '\\.', '')) %>%
  ggplot(aes(x = Geography, fill = Exited)) + geom_bar(position="dodge") + theme_bw() + coord_flip()

data %>% 
  mutate(Exited = str_replace(Exited, '\\.', '')) %>%
  ggplot(aes(x = HasCrCard, fill = Exited)) + geom_bar(position="dodge") + theme_bw() + coord_flip()


ggplot(data = data) +
  aes(x=Geography, fill=Exited) +
  geom_bar() + 
  theme_bw()


# Tenure
count(data, Tenure)

max(data$Tenure)
min(data$Tenure)

# counts for Customers who has credit card
count(data, HasCrCard)

# counts for Balance
count(data, Balance)

# counts for Credit score
count(data, CreditScore)

count(data, Gender)

# Data processing ------------------------------------------------------

### Cleaning the dataset ###

#we decided to create binary variable:
#	0 -(<= 650)  people with poor credit score,
#	1- (>650) - people with people with good and excellent credit score.

data$CreditScore_num <- ifelse(data$CreditScore <= 650, 1, 0)

round(data$Balance, digits = 0)
data$Balance_num <- ifelse(data$Balance == 0, 0, 1)

data["CreditScore_num"] = as.numeric(data$CreditScore_num)
data["Balance_num"] = as.numeric(data$Balance_num)

# counts for binary Credit Score and Balance
count(data, CreditScore_num)
count(data, Balance_num)

# checking for missing values
colSums(is.na(data))

data = data[-3]
View(data)


### Merge underrepresented classes ###


count(data, Geography) %>% arrange(-n)

data <- data %>% 
  mutate(
    Geography = ifelse(Geography %in% c('France'), 
                       Geography, 'other')
  )


# Analysis ----------------------------------------------------------------

# Linear Probability model
lpm = lm(Exited ~ ., data=data)
summary(lpm)
#interpret additional....

lpm.residuals = lpm$residuals
bptest(lpm.residuals ~., data=data)
#Breusch-Pagan test shows that there is no heteroskedasticity in our data.
#Following tests will provide better understandings.

#specification test 
resettest(lpm, power=2:3, type="fitted")
#p value < 5% we regect the HO that we have good model

#however result of linear probability test does not reflect reality
#we gonna use instead logit and probit model


# probit model
myprobit <- glm(Exited ~Age + Gender + Tenure + Balance + Balance_num + NumOfProducts + HasCrCard +
                  CreditScore +CreditScore_num + Geography  + IsActiveMember + EstimatedSalary, data = data, 
                family=binomial(link="probit"))
summary(myprobit)
# we can see that not all variables are significant
#we gonna use Likelihood ratio test to deal with it
myprobit_restricted <- glm(Exited ~Age + Gender +Balance + Balance_num 
                  + Geography  + IsActiveMember, data = data, 
                family=binomial(link="probit"))
summary(myprobit_restricted)
lrtest(myprobit, myprobit_restricted)
#p value = 0,06 > 5 % significant level  we cannot regect H0 which means that 
#our restricted model is more preferable 

# logit model
mylogit <- glm(Exited~Age + Gender + Tenure + Balance + Balance_num + NumOfProducts + HasCrCard +
                 CreditScore +CreditScore_num + Geography  + IsActiveMember + EstimatedSalary, data = data, 
               family=binomial(link="logit"))
summary(mylogit)

mylogit_restricted <- glm(Exited ~Age + Gender + Balance_num 
                           + Geography  + IsActiveMember, data = data, 
                           family=binomial(link="logit"))
summary(mylogit_restricted)
lrtest(mylogit, mylogit_restricted)
#p value = 0,058 > 5 % significant level  we cannot regect H0 which means that 
#our restricted model is more preferable

# Based on AIC information criteria we should choose logit model.
#  variables Age, Gender, Balance_num, Geography and IsActiveMember in our model are statistically significant at 5% significance level.

logit_non_linear = glm(Exited ~Age + Gender + Balance + Balance_num + Geography  + IsActiveMember + I(Age^2) + Gender*Geography,
                       data, family=binomial(link="logit"))

summary(logit_non_linear)


#Comparing logit model with interaction and variable to power and simple logit model, 
#second model will be chosen based on AIC information criteria

# creating quality table
library(stargazer)
stargazer(myprobit_restricted, mylogit_restricted, logit_non_linear, lpm, type="text")


#Marginal effects for the average observation
marr_effects=logitmfx( formula = Exited ~Age + Gender + Balance_num + Geography  + IsActiveMember + I(Age^2) + Gender*Geography,
                       data, atmean = T)

class(marr_effects)



# Interpretation of R^2
PseudoR2(logit_non_linear, which = "all")


# adjusted R^2
blr_rsq_adj_count(logit_non_linear) # function from package blorr

#count R^2
blr_rsq_count(logit_non_linear) # function from package blorr


# Testing hypothesis

# H0: Credit score = 0
# H1: Credit score != 0

logit_unrestricted = glm(Exited ~Age + Gender + Tenure + Balance + Balance_num + NumOfProducts + HasCrCard +
                           CreditScore +CreditScore_num + Geography  + IsActiveMember + EstimatedSalary + I(Age^2) + Gender*Geography,
                         data, family=binomial(link="logit"))

# removing Credit Score  variable from the model
logit_restricted = glm(Exited ~Age + Gender + Tenure + Balance + Balance_num + NumOfProducts + HasCrCard +
                          + Geography + Geography + CreditScore_num + IsActiveMember + EstimatedSalary + I(Age^2) + Gender*Geography,
                       data, family=binomial(link="logit"))

lrtest(logit_unrestricted, logit_restricted)
#p value = 0,2856 > 5 % significant level  we cannot regect H0 which means that 
#our restricted model is more preferable

# H0: Balance = 0
# H1: Balance != 0

logit_unrestricted1 = glm(Exited ~Age + Gender + Tenure + Balance + Balance_num + NumOfProducts + HasCrCard +
                            CreditScore +CreditScore_num + Geography  + IsActiveMember + EstimatedSalary + I(Age^2) + Gender*Geography,
                          data, family=binomial(link="logit"))

# removing Balance variable from the model
logit_restricted1 = glm(Exited ~Age + Gender + Tenure + NumOfProducts + HasCrCard +
                         CreditScore +CreditScore_num + Geography  + IsActiveMember + EstimatedSalary + I(Age^2) + Gender*Geography,
                        data, family=binomial(link="logit"))

lrtest(logit_unrestricted1, logit_restricted1) 
#p value = 0 < 5 % significant level  we  regect H0 which means that 
#our restricted model is less preferable

# H0: Gender = 0
# H1: Gender != 0

logit_unrestricted2 = glm(Exited ~Age + Gender + Tenure + Balance + Balance_num + NumOfProducts + HasCrCard +
                            CreditScore +CreditScore_num + Geography  + IsActiveMember + EstimatedSalary + I(Age^2) + Gender*Geography,
                          data, family=binomial(link="logit"))

# removing Gender variable from the model
logit_restricted2 = glm(Exited ~Age + Tenure + Balance + Balance_num + NumOfProducts + HasCrCard +
                          CreditScore +CreditScore_num + Geography  + IsActiveMember + EstimatedSalary + I(Age^2) + Gender*Geography,
                        data, family=binomial(link="logit"))

lrtest(logit_unrestricted2, logit_restricted2) 
#p value = 0 < 5 % significant level  we  regect H0 which means that 
#our restricted model is less preferable

# Hosmer-Lemeshow test

# H0: Model has no omitted variables
# H1: Model has omitted variables

gof.results = gof(logit_non_linear)
gof.results$gof

# Link Test

source("linktest.R")
linktest_results = linktest(logit_non_linear)
summary(linktest_results)

#quality table
stargazer(logit_non_linear, type = "text",  title="Logit non-linear Results", out = "table1.txt")



