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

## Assembled on 12 September 2026

No simulator screen capture exists for this episode (the campaigns ran headless), so the
synchronised view was rendered from the episode's own immutable bag by
`scripts/render_episode_replay.py`: map, ground-truth path and pose, AMCL estimate, current
Nav2 plan, laser returns and the injection/warning/event state, one frame per storyboard
frame at the same simulation times. Outputs: `sim.mp4` (replay), `risk_trace.mp4`,
`demo_90s.mp4` (title card, stacked replay over risk trace, closing card; 90.5 s).
ffmpeg: static 7.0.2 build from the imageio-ffmpeg package, linked at `~/.local/bin/ffmpeg`.
