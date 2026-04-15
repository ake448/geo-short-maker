# Digital Twin Capture Scaffold

This scaffold adds a headless capture path built on the existing interactive Earth Digital Twin viewer.

## Files

- `digital_twin_shots.js` — reusable shot presets and interpolated timelines
- `render_digital_twin.js` — headless frame capture runner for those presets

## Available shots

- `orbit`
- `flyover`
- `zoom`
- `close_oblique`
- `gtazoom`
- `skyline_descend`
- `night_city`
- `region_reveal`

## Example dry run

```powershell
Set-Location C:\broll-api\geography
node render_digital_twin.js --shot orbit --lat 33.7490 --lon -84.3880 --duration 3 --dry-run
```

## Example frame capture

```powershell
Set-Location C:\broll-api\geography
node render_digital_twin.js --shot skyline_descend --lat 33.7490 --lon -84.3880 --duration 2 --fps 12 --out-dir .\runs\_twin_test
```

## Intended future wiring

- map `3d_orbit` to `orbit`
- map `3d_flyover` to `flyover`
- map `3d_zoom` to `zoom`
- map `3d_close_oblique` to `close_oblique`
- map `3d_gtazoom` to `gtazoom` or `skyline_descend`

The capture script writes a `manifest.json` beside the generated frames so the pipeline can later decide how to encode them into video.
