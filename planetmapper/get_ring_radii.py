import urllib.request


def load_webpage(url: str) -> str:
    user_agent = {'User-agent': 'Mozilla/5.0'}
    request = urllib.request.Request(url, headers=user_agent)
    webpage = urllib.request.urlopen(request, timeout=31).read().decode()
    return webpage


def parse_ring_webpage(url: str) -> list[tuple[str, float]]:
    html = load_webpage(url)
    table = html.split('<table')[1].split('</table>')[0]
    rows = table.split('<tr>')[2:]
    rings: list[tuple[str, float]] = []
    for row in rows:
        name = row.split('<th>')[1].split('</th>')[0]
        cells = [c.split('</td>')[0].lstrip('>') for c in row.split('<td')[1:]]
        try:
            radius = float(cells[0].replace(',', '').lstrip('~'))
        except ValueError:
            radius = float('nan')
        rings.append((name, radius))
    return rings


url = 'https://nssdc.gsfc.nasa.gov/planetary/factsheet/jupringfact.html'

rings = parse_ring_webpage(url)
print(rings)
