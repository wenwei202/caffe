A=mmread('coo_conv2.weight');
mmwrite('coo_conv2_1.weight', A(1:128, :));
mmwrite('coo_conv2_1.weight', A(129:256, :));

A=mmread('coo_conv4.weight');
mmwrite('coo_conv4_1.weight', A(1:192, :));
mmwrite('coo_conv4_2.weight', A(193:384, :));

A=mmread('coo_conv5.weight');
mmwrite('coo_conv5_1.weight', A(1:128, :));
mmwrite('coo_conv5_2.weight', A(129:256, :));
