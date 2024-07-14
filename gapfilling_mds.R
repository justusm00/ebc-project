library(REddyProc)
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
#+++ Add time stamp in POSIX time format
EddyDataWithPosix <- EddyData %>% 
  filterLongRuns("NEE") %>% 
  fConvertTimeToPosix('YDH', Year = 'Year', Day = 'DoY', Hour = 'Hour')
#+++ Initalize R5 reference class sEddyProc for processing of eddy data
#+++ with all variables needed for processing later
EProc <- sEddyProc$new(
  'DE-Tha', EddyDataWithPosix, c('NEE','H','LE','Rg','Tair','VPD', 'Ustar'))
#Location of DE-Göttingen
EProc$sSetLocationInfo(LatDeg = 51.5, LongDeg = 10, TimeZoneHour = 1)  
#
#++ Fill NEE gaps with MDS gap filling algorithm (without prior ustar filtering)
EProc$sMDSGapFill('H', FillAll = FALSE)#


#++ Export gap filled and partitioned data to standard data frame
FilledEddyData <- EProc$sExportResults()


FilledEddyData

