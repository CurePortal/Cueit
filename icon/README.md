# CueIt icon

`cueit.svg` is the master icon source. Export platform assets with:

```bash
for s in 16 24 32 48 64 128 256 512; do
  rsvg-convert -w $s -h $s cueit.svg -o ../cueit/assets/cueit_${s}.png
done
magick ../cueit/assets/cueit_256.png ../cueit/assets/cueit.ico
python -c "from PIL import Image; Image.open('../cueit/assets/cueit_512.png').save('../cueit/assets/cueit.icns', format='ICNS')"
```
