#################################
# R Coursework Project 2021/2022
#################################

# Course name and code: Programming for Data Science ST2195
# Student UOL number: 180296505

# ====== Set working directory ====== #

# Set working directory
setwd("C:/Users/amand/Documents/SIM stuff/Course related/Year 4/ST2195 Programming for Data Science/Coursework By 1st Apr 2022/Data Expo Harvard/dataverse_files")

# Check working directory
getwd()

# Set memory limit to 5GB to reduce memory errors
memory.limit(size=50000)

# ============ Load libraries ============ #

# Replace library with install.packages("nameofpackage") to install packages for first-time users
library(readr) # Data manipulation
library(dplyr) # Alternatively, install.packages("tidyverse")
library(ggplot2) # Data visualization
library(igraph) # Network visualization
library(mlr3) # Machine learning
library(mlr3learners)
library(mlr3pipelines)
library(mlr3tuning)
library(xgboost)
library(ranger)
library(e1071)
library(mlr3viz)

# ====== Import CSV files ====== #

# Load in years 2005 & 2006 flight data from CSV files
# Note: TRUE is in all capital letters
flights2005 <- read.csv("2005.csv", header = TRUE)
flights2006 <- read.csv("2006.csv", header = TRUE)

# Concatenate both data sets into one data frame
flights <- rbind(flights2005, flights2006)
str(flights)

# ====== Data Cleaning ====== #

# Convert blank cells to NA
flights[flights == ""] <- NA

# Check for percentage of missing values
colMeans(is.na(flights))

# Drop irrelevant columns
# The '-' sign indicates dropping of variables
# Make sure the variable names do not have quotations when using subset() function
flights <- subset(flights, select = -c(FlightNum, TaxiIn, TaxiOut, Cancelled, CancellationCode, Diverted))

# Filling missing values with mean values
flights$DepTime[is.na(flights$DepTime)] <- mean(flights$DepTime, na.rm = TRUE)
flights$ArrTime[is.na(flights$ArrTime)] <- mean(flights$ArrTime, na.rm = TRUE)
flights$AirTime[is.na(flights$AirTime)] <- mean(flights$AirTime, na.rm = TRUE)

# Filling missing values with 0
flights <- flights %>% mutate(CRSDepTime = ifelse(is.na(CRSDepTime), 0, CRSDepTime),
                              CRSArrTime = ifelse(is.na(CRSArrTime), 0, CRSArrTime),
                              ActualElapsedTime = ifelse(is.na(ActualElapsedTime), 0, ActualElapsedTime),
                              CRSElapsedTime = ifelse(is.na(CRSElapsedTime), 0, CRSElapsedTime),
                              Distance = ifelse(is.na(Distance), 0, Distance),
                              CarrierDelay = ifelse(is.na(CarrierDelay), 0, CarrierDelay),
                              WeatherDelay = ifelse(is.na(WeatherDelay), 0, WeatherDelay),
                              NASDelay = ifelse(is.na(NASDelay), 0, NASDelay),
                              SecurityDelay = ifelse(is.na(SecurityDelay), 0, SecurityDelay),
                              LateAircraftDelay = ifelse(is.na(LateAircraftDelay), 0, LateAircraftDelay))

# Replace missing values for ArrDelay column with ArrTime subtracted by the CRSArrTime
flights$ArrDelay = ifelse(is.na(flights$ArrDelay) == TRUE, flights$ArrTime - flights$CRSArrTime, flights$ArrDelay)

# Replace missing values for DepDelay column by adding all the delay values together
flights$DepDelay = ifelse(is.na(flights$DepDelay) == TRUE, flights$CarrierDelay + flights$WeatherDelay + flights$NASDelay + 
                            flights$SecurityDelay + flights$LateAircraftDelay, flights$DepDelay)

# Create 'DepStatus' and 'ArrStatus' columns with status On-Time and Delayed for departures and arrivals
flights <- flights %>% mutate (DepStatus =
                                 case_when(DepDelay < 15 ~ "On-Time",
                                           DepDelay >= 15 ~ "Delayed"))

flights <- flights %>% mutate (ArrStatus =
                                 case_when(ArrDelay < 15 ~ "On-Time",
                                           ArrDelay >= 15 ~ "Delayed"))

# Replace negative values (early flights) with 0
flights[flights < 0] <- 0

# Coerce relevant columns to factor type
flights$Year = as.factor(flights$Year)
flights$DayofMonth = as.factor(flights$DayofMonth)
flights$UniqueCarrier = as.factor(flights$UniqueCarrier)
flights$TailNum = as.factor(flights$TailNum)
flights$Origin = as.factor(flights$Origin)
flights$Dest = as.factor(flights$Dest)

# Set levels for specific columns and coerce to factor type
flights$Month = as.factor(x = flights$Month)
levels(flights$Month) = c("January", "February", "March", "April", "May", "June", 
                          "July", "August", "September", "October", "November", "December")

flights$DayOfWeek = as.factor(flights$DayOfWeek)
levels(flights$DayOfWeek) = c("Monday", "Tuesday", "Wednesday", 
                              "Thursday", "Friday", "Saturday", "Sunday")

flights$DepStatus = as.factor(flights$DepStatus)
levels(flights$DepStatus) = c("On-Time", "Delayed")
flights$ArrStatus = as.factor(flights$ArrStatus)
levels(flights$ArrStatus) = c("On-Time", "Delayed")

# Convert DepTime and ArrTime to hours and drop values more than 24 hours
flights$DepTime = floor(flights$DepTime/100)
flights <- subset(flights, DepTime < 24)

flights$ArrTime = floor(flights$ArrTime/100)
flights <- subset(flights, ArrTime < 24)

# Round down AirTime minutes to nearest whole number
flights$AirTime = floor(flights$AirTime)

# Coerce DepTime and ArrTime to factor type
flights$DepTime = as.factor(flights$DepTime)
flights$ArrTime = as.factor(flights$ArrTime)

str(flights)

# Drop rows containing NAs
flights <- na.omit(flights)

# Locate any duplicate rows
distinct(flights)

# Check summary of data frame
summary(flights)

# =========== Exploratory data analysis =========== #

# Subset huge data set to 100k random observations for plotting
flightsplot <- sample_n(flights, 100000)

# Create scatter plot depicting the bi-variate correlation between DepTime and ArrTime
th <- theme_light()

ggplot(flightsplot, aes(x = DepTime, y = ArrTime)) +
  geom_point(alpha = 0.5) +
  th +
  theme(plot.title = element_text(size = 12)) +
  labs(x = "Departure Time (local hours)",
       y = "Arrival Time (local hours)",
       title = "Correlation between Departure Time and Arrival Time")

# Create scatterplot depicting the bivariate correlation between DepDelay and LateAircraftDelay
ggplot(flightsplot, aes(x = DepDelay, y = LateAircraftDelay)) +
  geom_point(alpha = 0.2) +
  th +
  theme(plot.title = element_text(size = 12)) +
  labs(x = "Departure Delay (local minutes)",
       y = "Late Aircraft Delay (local minutes)",
       title = "Correlation between Departure Delay and Late Aircraft Delay")
ggsave("DepDelay VS LateAircraftDelay_R.png")

# Correlation between DepDelay and ArrDelay
cor(flightsplot[c("DepDelay", "ArrDelay")], use = "pairwise.complete.obs")

# Create pair plot depicting the bivariate correlation between DepDelay and ArrDelay
pairs(flightsplot[c("DepDelay", "ArrDelay")])
ggsave("DepDelay VS ArrDelay_R.png")

# =========== Answering questions =========== #

# ================================================================================================ #
#Q1 When is the best time of day, day of the week, and time of year to fly to minimize delays?
# ================================================================================================ #

# Best time to fly to minimize delays

# Calculate the mean and average number of departures by day of the month
flightsplot %>%
  group_by(DayofMonth) %>%
  summarise(day_mean = floor(mean(DepDelay))) %>%
  # Create bar chart to portray the average departure delays on a daily basis
  ggplot(aes(x = DayofMonth, y = day_mean, fill = DayofMonth)) +
  geom_col() +
  th +
  theme(plot.title = element_text(size = 12),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  labs(y = "Mean Departure Delays (minutes)",
       title = "Average departure delays on a daily") +
  scale_x_discrete("Day", breaks = seq(0, 31, by = 5))

# Best hour to fly to minimize delays

# Calculate the mean and average number of departures by time of day
flightsplot %>%
  group_by(DepTime) %>%
  summarise(departure_time_mean = floor(mean(DepDelay))) %>%
  # Create line plot to portray the average departure delays per hour
  ggplot(aes(x = DepTime, y = departure_time_mean, group = 1)) +
  geom_line(color = "blue") +
  geom_point() +
  th +
  theme(plot.title = element_text(size = 12)) +
  labs(x = "Hour",
       y = "Mean Departure Delays (minutes)",
       title = "Average departure delays per hour") 

# Best day of the week to fly to minimize delays

# Calculate the mean and average number of departures by weeks
flightsplot %>%
  group_by(DayOfWeek) %>%
  summarise(week_mean = floor(mean(DepDelay))) %>%
  # Create bar chart to portray the average departure delays by the week
  ggplot(aes(x = DayOfWeek, y = week_mean, fill = DayOfWeek)) +
  geom_col() +
  th +
  theme(plot.title = element_text(size = 12),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  labs(x = "Week",
       y = "Mean Departure Delays (minutes)",
       title = "Average departure delays weekly")

# Best time of the year to fly to minimize delays

# Calculate the mean and average number of departures by month
flightsplot %>%
  group_by(Month) %>%
  summarise(month_mean = floor(mean(DepDelay))) %>%
  # Create bar chart to portray the average departure delays monthly
  ggplot(aes(x = Month, y = month_mean, fill = Month)) +
  geom_col() +
  th +
  theme(plot.title = element_text(size = 12), axis.text.x = element_text(angle = 90),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  labs(x = "Month",
       y = "Mean Departure Delays (minutes)",
       title = "Average departure delays monthly") +
  scale_x_discrete("Month")

# ========================================= #
# Q2 Do older planes suffer more delays?
# ========================================= #

# Load planes-data CSV file to data frame
planes <- read.csv("plane-data.csv", header = TRUE)

# Convert blank cells in planes to NA and drop rows with missing values
planes[planes == ""] <- NA
planes <- na.omit(planes)

# Using dplyr, inner join flights with planes where TailNum and tailnum are the keys
flights_merged <-  inner_join(flights,
                              select(planes, tailnum, plane_year = year),
                              by = c("TailNum" = "tailnum"))
# Coerce year column from str to int type
flights_merged$plane_year <- as.numeric(as.character(flights_merged$plane_year))
flights_merged$Year <- as.numeric(as.character(flights_merged$Year))

# Calculate the age of plane
flights_merged$planesAge <- (flights_merged$Year - flights_merged$plane_year)

# On average, planes retire at 25 years old
# Hence only keep rows with age equals or less than 25 and not less than 0
planes_age_delayed <- subset(flights_merged, planesAge <= 25 & planesAge > 0)

# Calculate the mean values of flight arrival delays based on plane age

planes_age_delayed %>%
  group_by(planesAge) %>%
  summarise(arr_delay_mean = mean(ArrDelay)) %>%
  # Create a scatter plot to show the relation between the two variables
  ggplot(aes(x = planesAge, y = arr_delay_mean, group = 1)) +
  geom_line(color = "blue") +
  geom_point() +
  th +
  theme(plot.title = element_text(size = 12),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  labs(title = "Relation between plane age and average arrival delays") +
  scale_x_continuous("Age of plane (years)", breaks = seq(0, 30, by = 10)) +
  scale_y_continuous("Mean Arrival Delays (minutes)")

# ======================================================================================== #
# Q3 How does the number of people flying between different locations change over time?
# ======================================================================================== #

# Return arrivals at top 10 destinations in two years
destinations <- flights %>%
  count(Dest) %>%
  top_n(10) %>%
  arrange(n, Dest) %>%
  mutate(Dest = factor(Dest, levels = unique(Dest)))

flights %>%
  filter(Dest %in% destinations$Dest) %>%
  mutate(Dest = factor(Dest, levels = levels(destinations$Dest))) %>%
  ggplot(aes(x = Dest, fill = Dest)) +
  geom_bar() +
  th +
  theme(plot.title = element_text(size = 12),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  labs(x = "Destination",
       title = "Most frequent destinations from years 2005 to 2006")

# Tabulate the number of flights flying from ATL to LAX in two years
ATL_LAX <- flights %>%
  filter(Origin == "ATL" & Dest == "LAX") %>%
  group_by(Month) %>%
  count(Month, name = "count")

# Tabulate the number of flights flying from ORD to IAH in two years
ORD_IAH <- flights %>%
  filter(Origin == "ORD" & Dest == "IAH") %>%
  group_by(Month) %>%
  count(Month, name = "count")

# Tabulate the number of flights flying from DFW to DEN in two years
DFW_DEN <- flights %>%
  filter(Origin == "DFW" & Dest == "DEN") %>%
  group_by(Month) %>%
  count(Month, name = "count")

# Tabulate the number of flights flying from PHX to LAS in two years
PHX_LAS <- flights %>%
  filter(Origin == "PHX" & Dest == "LAS") %>%
  group_by(Month) %>%
  count(Month, name = "count")

# Tabulate the number of flights flying from CVG to EWR in two years
CVG_EWR <- flights %>%
  filter(Origin == "CVG" & Dest == "EWR") %>%
  group_by(Month) %>%
  count(Month, name = "count")

# Add common key for all data frames
ATL_LAX$Routes <- "ATL to LAX"
ORD_IAH$Routes <- "ORD to IAH"
DFW_DEN$Routes <- "DFW to DEN"
PHX_LAS$Routes <- "PHX to LAS"
CVG_EWR$Routes <- "CVG to EWR"

# Combine all data frames into one data frame
routes_df <- rbind(ATL_LAX,ORD_IAH,DFW_DEN,PHX_LAS,CVG_EWR)

# Create line plots to show the different routes of flights flying between different locations over time
ggplot(routes_df, aes(x = Month, y = count, group = Routes, color = Routes)) + 
  geom_line() +
  geom_point() +
  th +
  theme(plot.title = element_text(size = 12), axis.text.x = element_text(angle = 90),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  labs(x = "Month",
       title = "Number of people flying between different locations over time")

# ========================================================================================== #
# Q4 Can you detect cascading failures as delays in one airport create delays in others?
# ========================================================================================== #

# Return the top 10 origin airports with the highest average departure delays
origin_delays <- flights %>%
  group_by(Origin) %>%
  filter(DepDelay >= 15) %>%
  summarise(origin_dep_mean = floor(mean(DepDelay, na.rm = TRUE)), top_origin = n()) %>%
  slice_max(top_origin, n = 10) %>%
  arrange(desc(top_origin))

origin_delays %>%
  ggplot(aes(x = Origin, y = origin_dep_mean, fill = Origin)) +
  geom_col() +
  coord_flip() +
  th +
  theme(plot.title = element_text(size = 11),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  labs(x = "Origin",
       y = "Mean Departure Delays (minutes)",
       title = "Top 10 origin airports with the highest departure delays over two years") 

# Return the top 10 destination airports with the highest average arrival delays
dest_delays <- flights %>%
  group_by(Dest) %>%
  filter(ArrDelay >= 15) %>%
  summarise(dest_dep_mean = floor(mean(ArrDelay, na.rm = TRUE)), top_dest = n()) %>%
  slice_max(top_dest, n = 10) %>%
  arrange(desc(top_dest))

dest_delays %>%
  ggplot(aes(x = Dest, y = dest_dep_mean, fill = Dest)) +
  geom_col() +
  coord_flip() +
  th +
  theme(plot.title = element_text(size = 11),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  labs(x = "Destination",
       y = "Mean Arrival Delays (minutes)",
       title = "Top 10 destination airports with the highest arrival delays over two years") 

# Combine origin_delays and dest_delays into a single data frame
airports_delayed <- cbind(origin_delays, dest_delays)

# Create edges for network graph
edges_df <- airports_delayed %>%
  select(Origin, Dest, origin_dep_mean) %>%
  na.omit() %>%
  group_by(Origin, Dest)

edges_df1 <- edges_df %>%
  arrange(desc(origin_dep_mean))

# View statistics of data frame
summary(edges_df1)

# Create histogram to show distribution of airports with the most flight departure delays
ggplot(edges_df1, aes(x = origin_dep_mean)) +
  geom_histogram(binwidth = 1,
                 colour = "black", fill = "white") +
  geom_vline(aes(xintercept = median(origin_dep_mean, na.rm = TRUE)),
             color = "red", linetype = "dashed", size = 1)

# Create box plot to show median and outliers of airports with the most flight departure delays
ggplot(edges_df1, aes(x = origin_dep_mean, y = origin_dep_mean)) +
  geom_boxplot(outlier.colour = "blue",
               outlier.shape = 1,
               outlier.size = 2,
               notch = FALSE)

# Plot network graph for delayed flight routes to show impact of departure delays more than 52 minutes (median)
edges_df1 %>%
  filter(origin_dep_mean > 52) %>%
  graph.data.frame() -> airportsdepgreaterthan52

plot(airportsdepgreaterthan52,
     layout = layout.fruchterman.reingold,
     vertex.color = "blue", vertex.size = 3,
     vertex.frame.color = "gray", vertex.label.color = "black",
     vertex.label.cex = 0.6, vertex.label.dist = 0.5,
     edge.curved = 0.4, edge.arrow.size = 0.2,
     edge.color = "light blue")

# Plot network graph for delayed flight routes to show impact of departure delays more than 60 minutes (1 hour)
edges_df1 %>%
  filter(origin_dep_mean >= 60) %>%
  graph.data.frame() -> airportsdepgreaterthan60

plot(airportsdepgreaterthan60,
     layout = layout.fruchterman.reingold,
     vertex.color = "blue", vertex.size = 3,
     vertex.frame.color = "gray", vertex.label.color = "black",
     vertex.label.cex = 0.6, vertex.label.dist = 0.5,
     edge.curved = 0.4, edge.arrow.size = 0.2,
     edge.color = "light blue")

# =========================================================================== #
# Q5 Use the available variables to construct a model that predicts delays.
# =========================================================================== #

# Subset data to drop irrelevant variables
dropcols <- c("CRSDepTime", "CRSArrTime", "UniqueCarrier",
              "CRSElapsedTime", "Dest", "Distance", "ArrStatus")
flights_samples_cols <- flights[dropcols]

# Rename ArrStatus variable to Delayed
flights_samples_cols <- rename(flights_samples_cols, c("Delayed" = "ArrStatus"))

# Convert Delayed variable to binary data type where 1 = Delayed, 0 = On-Time
flights_samples_cols$Delayed <- ifelse(flights_samples_cols$Delayed == "Delayed",1,0)
flights_samples_cols

# Coerce Delayed variable to factor type
flights_samples_cols$Delayed <- as.factor(flights_samples_cols$Delayed)

# Split flight data into 50,000 samples from 2005 and 50,000 samples from 2006
flights_samples_2005  <- head(flights_samples_cols, n = 50000)
flights_samples_2006  <- tail(flights_samples_cols, n = 50000)

# Combine both data sets into a single data frame
flights_samples <- rbind(flights_samples_2005, flights_samples_2006)

# Preview data set for machine learning
head(flights_samples)
str(flights_samples)
summary(flights_samples)

# Split training and test sets (80:20)
n <- nrow(flights_samples)

set.seed(100)
train_set <- sample(n, round(0.8*n))
test_set <- setdiff(1:n, train_set)

# Set up task
task <- TaskClassif$new("flights_samples", backend = flights_samples, target = "Delayed")
task$select(c("CRSDepTime", "CRSArrTime", "UniqueCarrier",
              "CRSElapsedTime", "Dest", "Distance"))
task

# Show all measures
msr()
measure <- msr("classif.ce")

# As some variables are factors,
# we need to convert them to numerical values through factor encoder for R machine learning pipelines
# method = "treatment": create n-1 columns leaving out the first factor level
# method = "one-hot": create a new column for each factor level
fencoder <- po("encode", method = "treatment",
               affect_columns = selector_type("factor"))

# Tuning hyperparameters
tuner <- tnr("grid_search")
terminator <- trm("evals", n_evals = 20)

# Logistic regression pipelines
learner_lr <- lrn("classif.log_reg")

# Note %>>% is used for R pipelines
gc_lr <- po("imputemean", affect_columns = selector_type("numeric")) %>>%
  po("imputemode", affect_columns = selector_type(c("factor"))) %>>%
  po(learner_lr)

glrn_lr <- GraphLearner$new(gc_lr)

glrn_lr$train(task, row_ids = train_set)
glrn_lr$predict(task, row_ids = test_set)$score()

# Gradient boosting classifier pipelines
set.seed(42)

learner_gbc <- lrn("classif.xgboost")

gc_gbc <- po("imputemean", affect_columns=selector_type("numeric")) %>>%
  po("imputemode", affect_columns = selector_type(c("factor"))) %>>%
  fencoder %>>%
  po(learner_gbc)

glrn_gbc <- GraphLearner$new(gc_gbc)

glrn_gbc$train(task, row_ids = train_set)
glrn_gbc$predict(task, row_ids = test_set)$score()

# Random forest classifier pipelines
set.seed(42)

learner_rfc <- lrn("classif.ranger") 
learner_rfc$param_set$values <- list(min.node.size = 4)

gc_rfc <- po("imputemean", affect_columns = selector_type("numeric")) %>>%
  po("imputemode", affect_columns = selector_type(c("factor"))) %>>%
  po(learner_rfc)

glrn_rfc <- GraphLearner$new(gc_rfc)

tune_ntrees <- ParamSet$new (list(
  ParamInt$new("classif.ranger.num.trees", lower = 50, upper = 600)
))

at_rfc <- AutoTuner$new(
  learner = glrn_rfc,
  resampling = rsmp("cv", folds = 3),
  measure = measure,
  search_space = tune_ntrees,
  terminator = terminator,
  tuner = tuner
)

at_rfc$train(task, row_ids = train_set)
at_rfc$predict(task, row_ids = test_set)$score()

# Benchmarking -- Compare results of different learners
set.seed(1) # For reproducible results

# List of learners
lrn_list <- list(
  glrn_lr,
  glrn_gbc,
  at_rfc
)

# Set the benchmark design and run the comparisons
bm_design <- benchmark_grid(task = task, resamplings = rsmp("cv", folds=3), 
                            learners = lrn_list)
bmr <- benchmark(bm_design, store_models = TRUE)

# Visualize comparisons with box plot
autoplot(bmr) + theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Print overall measure for each classification model
bmr$aggregate(measure)