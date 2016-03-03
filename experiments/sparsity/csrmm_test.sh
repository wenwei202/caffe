#!/bin/bash

# first argument: image size
# second argument: # of input channels
# third argument: image size
# fourth argument: stride
# fifth argument: pad
./csrmm_test conv1.mtx 3025 3 227 4 0
echo
./csrmm_test conv2_1.mtx 729 48 27 1 2
echo
./csrmm_test conv2_2.mtx 729 48 27 1 2
echo
./csrmm_test conv3.mtx 169 256 13 1 1
echo
./csrmm_test conv4_1.mtx 169 192 13 1 1
echo
./csrmm_test conv4_2.mtx 169 192 13 1 1
echo
./csrmm_test conv5_1.mtx 169 192 13 1 1
echo
./csrmm_test conv5_2.mtx 169 192 13 1 1
