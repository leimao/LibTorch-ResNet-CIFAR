#!/bin/bash

wget -N https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

if md5sum --status -c cifar-10-binary.tar.gz.md5; then
    # The MD5 sum matched
    echo "CIFAR10 dataset downloaded is intact."
else
    # The MD5 sum didn't match
    echo "CIFAR10 dataset downloaded is problematic."
fi

tar xzf cifar-10-binary.tar.gz



