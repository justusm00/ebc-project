library(REddyProc)
library(base)
# Load necessary libraries
library(dplyr)
library(tidyr)
library(lubridate)


#### this script is used to compute the MSE for the predictions of the MDS algorithm on the artificial gap data
#### this must be done site-wise for BG and GW


######### auxiliary functions

# Function to create a complete sequence of dates and times
create_complete_sequence <- function(start_date, end_date) {
  seq(from = start_date, to = end_date, by = "30 min")
}


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


gap_filling_mds <- function(df, loc){
  # filter for location
  df <- subset(df, location == loc)
  # Find the range of dates in your dataframe
  start_date <- ymd_hms(paste(df$year[1], df$month[1], df$day[1], floor(df$hour[1]), (df$hour[1] - floor(df$hour[1])) * 60, "00", sep = "-"))
  end_date <- ymd_hms(paste(tail(df$year, 1), tail(df$month, 1), tail(df$day, 1), floor(tail(df$hour, 1)), (tail(df$hour, 1) - floor(tail(df$hour, 1))) * 60, "00", sep = "-"))
  
  
  # Generate the complete sequence of date-times
  complete_sequence <- create_complete_sequence(start_date, end_date)
  
  
  # Create a dataframe from the complete sequence
  complete_df <- data.frame(datetime = complete_sequence)
  complete_df <- complete_df %>%
    mutate(
      year = year(datetime),
      month = month(datetime),
      day_of_year = yday(datetime),
      hour = hour(datetime) + minute(datetime) / 60
    ) %>%
    select(-datetime)
  
  complete_df
  
  # Merge with the original dataframe
  merged_df <- complete_df %>%
    left_join(df, by = c("year", "month", "day_of_year", "hour"))
  
  
  
  old_names <- c("year", "day_of_year", "hour",
                 "NEE_orig", "LE_orig", "H_orig", "incomingShortwaveRadiation",
                 "airTemperature", "soilTemperature", "relativeHumidity", "waterPressureDeficit", "Ustar", "artificial_gap")
  
  
  
  df <- merged_df[, old_names]
  new_names <- c("Year", "DoY", "Hour", "NEE", "LE", "H", "Rg", "Tair", "Tsoil", "rH", "VPD", "Ustar", "artificial_gap")
  names(df) <- new_names
  df_orig <- df
  
  
  # set original values to na for artificial gaps
  df$H[df$artificial_gap != 0] <- NA
  df$LE[df$artificial_gap != 0] <- NA
  
  
  
  #+++ If not provided, calculate VPD from Tair and rH
  df$VPD <- fCalcVPDfromRHandTair(df$rH, df$Tair)
  
  
  
  df_orig$H_f <- processEddyData(df, "H")
  df_orig$LE_f <- processEddyData(df, "LE")
  
  
  

  df_comp <- df_orig[complete.cases(df_orig[, c("H", "LE")]), ]
  
  return(df_comp)
}





########## gap filling

# load data
data <- read.csv("data/preprocessed/data_merged_with_nans.csv")

# create hour column
data$hour <- data$X30min * 0.5 
data <- subset(data, select = -X30min)


df_comp_bg <- gap_filling_mds(data, 0)
df_comp_bg$location <- 0

df_comp_gw <- gap_filling_mds(data, 1)
df_comp_gw$location <- 1


df_combined <- rbind(df_comp_bg, df_comp_gw)

# select only subset


df_combined$X30min <- as.numeric(df_combined$Hour * 2)
df_combined <- df_combined[, c("Year", "DoY", "location", "X30min", "H_f", "LE_f")]


names(df_combined) <- c("year", "day_of_year", "location", "X30min", "H_f", "LE_f")



write.csv(df_combined, file = "data/gapfilled/combined_gapfilled_mds.csv", row.names = FALSE)

