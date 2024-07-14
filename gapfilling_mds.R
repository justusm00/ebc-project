library(REddyProc)
EddyData <- read.csv("data/preprocessed/data_merged_with_nans.csv")
data <- Example_DETha98
data
EddyData <- EddyData[, c("year", "day_of_year", "X30min",
                         "CO2", "LE_orig", "H_orig", "incomingShortwaveRadiation",
                         "airTemperature", "")]