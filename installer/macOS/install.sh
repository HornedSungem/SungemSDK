#!/bin/bash

set -e

echo "****** INSTALLATION START ******"

#----------------------------------------

which -s brew
if [[ $? != 0 ]] ; then
	/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
fi

#----------------------------------------

brew update
brew install libusb

#----------------------------------------

echo "****** INSTALLATION COMPLETE ******"
