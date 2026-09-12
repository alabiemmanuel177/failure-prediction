# Demo assembly for f99d2bf0-48dd-4853-89b1-2077b02af73b

ffmpeg detected: no; install ffmpeg and rsvg-convert.

Rasterise the SVG frames, then assemble at the storyboard frame rate and mux with the
simulator screen capture (`sim.mp4`, trimmed to the same episode span):

```bash
for f in frames/frame_*.svg; do rsvg-convert -w 1000 -h 450 "$f" -o "${f%.svg}.png"; done
ffmpeg -framerate 2 -i frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p risk_trace.mp4
ffmpeg -i sim.mp4 -i risk_trace.mp4 -filter_complex "[0:v][1:v]vstack=inputs=2" -c:v libx264 demo_90s.mp4
```

This command is documented only; the storyboard script never executes it.
