#!/usr/bin/env python3

import sys
import os

pwd = os.path.abspath(os.path.join(os.path.basename(os.path.dirname(__file__)), '../'))
os.system("cd '%s'; make" % (pwd,))