# cat train.info | grep "Test net output #0: accuracy =" | awk '{print $11}'

import re
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--traininfo', type=str, required=True)
    args = parser.parse_args()
    traininfo = open(args.traininfo,'r')
    error = []
    for line in traininfo:
        if re.match(".*(Test net output \#0\: accuracy =).*", line):
            error.append(1-float(line.split()[10]))
    print error
    plt.subplot(121)
    plt.plot(error)
    plt.subplot(122)
    plt.plot( np.log10(error))
    #plt.ylim((0.1, 0.5))
    file = os.path.dirname(args.traininfo)+'/accu.png'
    plt.savefig(file)
    #plt.show()
