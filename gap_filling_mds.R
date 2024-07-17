library(REddyProc)
library(base)


# load data
EddyData <- read.csv("data/preprocessed/data_merged_with_nans.csv")


# filter for bg
EddyData <- subset(EddyData, location == 0)
# filter for 2024
EddyData <- subset(EddyData, year > 2023)





old_names <- c("year", "day_of_year", "X30min",
               "NEE_orig", "LE_orig", "H_orig", "incomingShortwaveRadiation",
               "airTemperature", "soilTemperature", "relativeHumidity", "waterPressureDeficit", "Ustar")



EddyData <- EddyData[, old_names]
new_names <- c("Year", "DoY", "Hour", "NEE", "LE", "H", "Rg", "Tair", "Tsoil", "rH", "VPD", "Ustar")
EddyData$X30min <- EddyData$X30min * 0.5 + 0.5
names(EddyData) <- new_names

#+++ If not provided, calculate VPD from Tair and rH
EddyData$VPD <- fCalcVPDfromRHandTair(EddyData$rH, EddyData$Tair)



processEddyData <- function(df, column_name) {
  
  # Convert time to POSIX format
  EddyDataWithPosix <- df %>%
    filterLongRuns(column_name) %>%
    fConvertTimeToPosix('YDH', Year = 'Year', Day = 'DoY', Hour = 'Hour')
  
  # Initialize sEddyProc for processing
  EProc <- sEddyProc$new(
    'DE-Goe', EddyDataWithPosix, c('NEE','H','LE','Rg','Tair','VPD', 'Ustar'))
  
  # Set location information
  EProc$sSetLocationInfo(LatDeg = 51.5, LongDeg = 10, TimeZoneHour = 1)
  
  # Fill gaps with MDS gap filling algorithm
  EProc$sMDSGapFill(column_name, FillAll = FALSE)
  
  # Export gap filled and partitioned data to standard data frame
  FilledEddyData <- EProc$sExportResults()
  
  return(FilledEddyData[[paste0(column_name, "_f")]])
}


compute_mse <- function(EddyData) {
  df <- EddyData
  df_copy <- df
  # Get indices of a random 20% of rows where H and LE are not NA ( this is the testset, everything else is the trainset)
  valid_indices <- which(!is.na(EddyData$H) &!is.na(EddyData$LE))
  sample_size <- ceiling(0.2 * length(valid_indices))
  sampled_indices <- sample(valid_indices, sample_size)
  
  # artificially set test values to NA
  df$H[sampled_indices] <- NA
  df$LE[sampled_indices] <- NA
  # fill gaps
  df_copy$H_f <- processEddyData(df, "H")
  df_copy$LE_f <- processEddyData(df, "LE")
  mse_h <- mean((df_copy[sampled_indices, ]$H - df_copy[sampled_indices, ]$H_f)**2)
  mse_le <- mean((df_copy[sampled_indices, ]$LE - df_copy[sampled_indices, ]$LE_f)**2)
  mse_total = mse_h + mse_le
  return(mse_total)
}



mses <- c()
for (i in 1:1000) {
  mses <- c(mses, compute_mse(EddyData))
}

mses
mean(mses)




