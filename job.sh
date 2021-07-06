#!/bin/bash

if [ `convert $1 -colorspace hsb -resize 1x1 txt:- | grep -o 'hsb\(.*,.*,.*\)' | grep -o '[0-9]*%)' | grep -o '[0-9]*'` -gt 20 ]; then
  echo $1
else
  rm "$1"
fi
