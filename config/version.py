import argparse
from datetime import datetime

def main(version, number, tag=None):
    if number:
        print(int(version.split('.')[-1]) + 1)
    else:
        datev = f'{datetime.now():%Y%m%d}'
        dayv = 0
        if datev in version:
            dayv = int(version.split('.')[-1]) + 1

        print(f'{tag + "." if tag else ""}{datev}.{dayv}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'Versionator',
                    description = 'Manages versions of things',
                    epilog = 'One version to rule them all, one version to find them, one version to bring them all and in the darkness bind them')
    parser.add_argument('-v', '--version', type=str, help='latest version', required=True)
    parser.add_argument('-t', '--tag', type=str, help='tag to prepend')
    parser.add_argument('-n', '--number', action='store_true')
    args = parser.parse_args()
    main(args.version, args.number, args.tag)