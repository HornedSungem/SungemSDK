#!/usr/bin/env python3
# coding=utf-8

import sys
import hsapi as hs
import getopt

def upgrade(file_path):
    device_names = hs.EnumerateDevices()
    if len(device_names) < 1:
        print("Error - No HS devices found.")
        quit()
    
    device = hs.Device(device_names[0])
    try:
        device.OpenDevice()
    except:
        print("Error - Cound not open HS device.")
        quit()

    print("HS Device opened normally.")

    try:
        device.UpdateApp(file_path)
    except:
        print("Error - Cound not upgrade HS device.")
        quit()

    print("Device upgrade successful.")


if __name__=="__main__":
    print("Firmware Upgrade")
    def usage():
        print("usage: %s -f <elf file>" % __file__)

    opts, args = getopt.getopt(sys.argv[1:], "f:", ["file="])
    elf_file = None

    for opt, arg in opts:
        if opt in ("-f", "--file"):
            elf_file = arg

    if elf_file is None:
        usage()
        sys.exit()

    sys.exit(upgrade(elf_file))