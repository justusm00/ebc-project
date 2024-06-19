Ground Heat Flux:
Determine thermal conductivity using relative moisture measurements, find value in literature [Van Wijk & de Vries]
Gradients between all measurements (5->15, 15->30, 5->30) -> mean of gradients -> Ground heat flux
-> Interpolation method: Diurnal means (Justus)


Incoming energy: 
incoming - outgoing shortwave radiation (ignore longwave since only shortwave from sun)
If net radiation in dataset, use that


Latent & Sensible Heat Flux:
Are included in the dataset, interpolation method?
-> Neural Net? -> Use data from 2023 for additional training data -> discuss bias because of changing climate
-> Data Cleaning -> Tag + Tageszeit codieren, NaNs entfernen, evtl. outlier & normalisieren (Dennis) 
-> Predict H_orig and LE_orig (Robin)
( -> Data Augmentation )
-> Compare to gap filling of original data and diurnal mean interpolation


Results:
-> Mean daily energy balance closure over timespan of data, month
-> Difference in balance closure between grass and forest (also over whole time & each month)
-> Look for days with extreme gap values -> inspect metheo data and discuss


( Bowen Ratio method:
-> If we can't get to 15 minutes in our presentation :D )