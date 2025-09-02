#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to scrape ring radii data from NASA Goddard planetary factsheets
https://nssdc.gsfc.nasa.gov/planetary/factsheet

The printed output from running this file can be used in rings.json
"""

import html
import json
import urllib.request


def main():
    print_json()


def print_json():
    ring_data = {}
    for planet, p in [
        ('jupiter', 'jup'),
        ('saturn', 'sat'),
        ('uranus', 'uran'),
        ('neptune', 'nep'),
    ]:
        url = f'https://nssdc.gsfc.nasa.gov/planetary/factsheet/{p}ringfact.html'
        rings = parse_ring_webpage(url)
        ring_data[planet.upper()] = rings

    json_string = json.dumps(ring_data, indent=4, ensure_ascii=False)
    print(json_string)


def parse_ring_webpage(url: str) -> dict[str, list[float]]:
    page_html = load_webpage(url)
    table = page_html.split('<table')[1].split('</table>')[0]
    rows = table.split('<tr>')[2:]
    rings_dict: dict[str, list[float]] = {}
    edges_dict = {}
    for row in rows:
        parse_row(row, rings_dict, edges_dict)
    for k, v in edges_dict.items():
        rings_dict[k] = [v['inner edge'], v['outer edge']]

    # Sort the rings
    out: dict[str, list[float]] = dict(
        sorted(rings_dict.items(), key=lambda x: x[1][0])
    )

    return out


def parse_row(
    row: str,
    rings_dict: dict[str, list[float]],
    edges_dict: dict[str, dict[str, float]],
):
    try:
        name = row.split('<th>')[1].split('</th>')[0]
        name = html.unescape(name).strip()
    except IndexError:
        return
    if name.casefold().endswith('equator'):
        return

    cells = [c.split('</td>')[0].lstrip('>') for c in row.split('<td')[1:]]
    radius_cell = cells[0].replace(',', '').lstrip('~')
    if '-' in radius_cell:
        radii = [float(s) for s in radius_cell.split('-')]
        rings_dict[name] = radii
        return
    try:
        radius = float(radius_cell)
    except ValueError:
        return
    for s in ('inner edge', 'outer edge'):
        if s in name:
            name = name.replace(s, '').strip()
            edges_dict.setdefault(name, {})[s] = radius
            return
    rings_dict[name] = [radius]


def load_webpage(url: str) -> str:
    # pylint: disable=consider-using-with
    user_agent = {'User-agent': 'Mozilla/5.0'}
    request = urllib.request.Request(url, headers=user_agent)
    webpage = urllib.request.urlopen(request, timeout=31).read().decode()
    return webpage


if __name__ == '__main__':
    main()
