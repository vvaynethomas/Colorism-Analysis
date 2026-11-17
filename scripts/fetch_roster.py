"""Rebuild data/executives.csv from Wikipedia's "List of chief executive officers".

    python scripts/fetch_roster.py [--out data/executives.csv]

The committed file is the August 2020 snapshot used for the results in the README; the
live page has changed since. Executive names are stored with spaces encoded as %20 so they
can be used directly as image-search queries and file names.
"""
import argparse
from io import StringIO

import pandas as pd
import requests

URL = 'https://en.wikipedia.org/wiki/List_of_chief_executive_officers'


def fetch_roster(url=URL):
    html = requests.get(url, headers={'User-Agent': 'colorism-analysis roster script'},
                        timeout=30).text
    df = pd.read_html(StringIO(html))[0][['Company', 'Executive', 'Title', 'Since']].copy()
    df['Executive'] = df['Executive'].str.replace(' ', '%20')
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='data/executives.csv')
    args = parser.parse_args()
    roster = fetch_roster()
    roster.to_csv(args.out, index=False)
    print(f'{len(roster)} executives written to {args.out}')
