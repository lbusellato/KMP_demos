#!/usr/bin/env python3

from kmp import UI

import logging
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
    datefmt='%Y-%m-%d,%H:%M:%S'
)

def main():
    ui = UI()
    input()

if __name__=='__main__':
    main()