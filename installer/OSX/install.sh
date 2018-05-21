#!/bin/bash

# Detect brew
which -s brew
if [[ $? != 0 ]] ; then
    # Install Homebrew
    /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
fi

brew update
brew install -y $(cat "../deps/requirements_brew.txt")
python3 -m pip install -r ../deps/requirements_osx.txt

