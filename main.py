#!/usr/bin/env python3

import logging
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
    datefmt='%Y-%m-%d,%H:%M:%S'
)
import sys

from kmp import demo1, demo2, demo3

def main(demo):
    if demo == '1':
        demo1()
    if demo == '2':
        demo2()
    if demo == '3':
        demo3()
    print("Press any key to exit...")
    input()

if __name__=='__main__':
    demo = sys.argv[-1] if len(sys.argv) > 1 else None
    main(demo)