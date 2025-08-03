#!/bin/bash

#Read in arguments
FNAME_LIST=$1
MAINDIR=$2



#Get fnames
file=$FNAME_LIST

i=0

while read fname imname; do

    if [ $i -lt 0 ]; then
        i=$((i+1))
        continue
    fi

    #Get output fnames
    FNAME_OUT="$MAINDIR"preprocess/stripe/"$imname"_noisechisel_mask.fits
    LOG_FNAME="$MAINDIR"preprocess/stripe/"$imname"_noisechisel.log    

    #Print input fname
    echo $imname

    #Remove old files
    rm $FNAME_OUT
    rm $LOG_FNAME

    #Run NoiseChisel
    astnoisechisel $fname -o $FNAME_OUT -h 1 >> $LOG_FNAME
    gzip --best $FNAME_OUT

    i=$((i+1))

done < $file

