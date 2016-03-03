#!/bin/bash

../../../csrmm_test coo_conv1.weight 3025 3 227 4 0
echo
../../../csrmm_test coo_conv2_1.weight 729 48 27 1 2
echo
../../../csrmm_test coo_conv2_2.weight 729 48 27 1 2
echo
../../../csrmm_test coo_conv3.weight 169 256 13 1 1
echo
../../../csrmm_test coo_conv4_1.weight 169 192 13 1 1
echo
../../../csrmm_test coo_conv4_2.weight 169 192 13 1 1
echo
../../../csrmm_test coo_conv5_1.weight 169 192 13 1 1
echo
../../../csrmm_test coo_conv5_2.weight 169 192 13 1 1
