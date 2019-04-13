#!/bin/bash

mkdir -p build
cd build

cmake ..
if [ $? != 0 ]
then
	echo "Error on cmake : exit -1"
	exit -1
fi

make
if [ $? != 0 ]
then
	echo "Error on make : exit -1"
	exit -1
fi

make run_all
if [ $? != 0 ]
then
	echo "Error on make run_all : exit -1"
	exit -1
fi

make plot_all
if [ $? != 0 ]
then
	echo "Error on make plot_all : exit -1"
	exit -1
fi



