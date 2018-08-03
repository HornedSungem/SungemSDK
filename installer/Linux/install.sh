#!/bin/bash

set -e

echo "****** INSTALLATION START ******"

#----------------------------------------

script_path=$(cd `dirname $0`; pwd)

sudo cp $script_path/../99-hornedsungem.rules /etc/udev/rules.d/
sudo chmod a+x /etc/udev/rules.d/99-hornedsungem.rules
sudo udevadm control --reload
sudo udevadm trigger

#----------------------------------------

echo "****** INSTALLATION COMPLETE ******"