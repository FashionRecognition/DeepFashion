# Extract all images from
import zipfile

archive = zipfile.ZipFile(r'C:\Users\mike_000\Downloads\img.zip')

for file in archive.namelist():
    print(file)
    if file.startswith('img/'):
        archive.extract(file, r'C:\Users\mike_000\Fashion')
