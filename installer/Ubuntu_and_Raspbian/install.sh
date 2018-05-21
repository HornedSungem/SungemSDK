#!/bin/bash

sudo apt-get install -y $(cat "../deps/requirements_apt.txt")

if [ "$1" = "tuna" ] ; then
	echo "Using TUNA mirror"
	python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r ../deps/requirements_linux.txt
else
	python3 -m pip install -r ../deps/requirements_linux.txt
fi

sudo sh install-opencv.sh

sudo cp 99-hornedsungem.rules /etc/udev/rules.d/
sudo chmod +x /etc/udev/rules.d/99-hornedsungem.rules
sudo udevadm control --reload

