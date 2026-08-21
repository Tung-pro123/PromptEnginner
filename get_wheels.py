import urllib.request
import re

urls = [
    'https://developer.download.nvidia.com/compute/redist/jp/v461/onnxruntime/',
    'https://developer.download.nvidia.com/compute/redist/jp/v45/onnxruntime/'
]

for url in urls:
    try:
        html = urllib.request.urlopen(url).read().decode('utf-8')
        wheels = re.findall(r'href="(.*?\.whl)"', html)
        print(f"URL: {url}")
        for w in wheels:
            print("  " + w)
    except Exception as e:
        print(f"Error reading {url}: {e}")
