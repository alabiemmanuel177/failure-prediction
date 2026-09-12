# Demo assembly for a75e4a6c-71b2-472a-9445-3edd06bcee14

ffmpeg detected: yes (/home/eao/.local/bin/ffmpeg).

Rasterise the SVG frames, then assemble at the storyboard frame rate and mux with the
simulator screen capture (`sim.mp4`, trimmed to the same episode span):

```bash
for f in frames/frame_*.svg; do rsvg-convert -w 1000 -h 450 "$f" -o "${f%.svg}.png"; done
ffmpeg -framerate 2 -i frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p risk_trace.mp4
ffmpeg -i sim.mp4 -i risk_trace.mp4 -filter_complex "[0:v][1:v]vstack=inputs=2" -c:v libx264 demo_90s.mp4
```

This command is documented only; the storyboard script never executes it.
