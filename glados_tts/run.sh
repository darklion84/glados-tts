#!/bin/sh
# Settings from /data/options.json (Home Assistant add-on options, or a mounted folder for plain Docker):
# threads, length_scale (speech tempo, higher = slower), debug. Missing file -> defaults.
set -eu
opt() { python -c "import json,sys; print(json.load(open('/data/options.json')).get(sys.argv[1], sys.argv[2]))" "$1" "$2" 2>/dev/null || echo "$2"; }
TTS_THREADS="$(opt threads 2)"
LENGTH_SCALE="$(opt length_scale 1.2)"
DEBUG_FLAG=""
[ "$(opt debug False)" = "True" ] && DEBUG_FLAG="--debug"
export TTS_THREADS
echo "glados-tts: threads=$TTS_THREADS length_scale=$LENGTH_SCALE debug=${DEBUG_FLAG:-off}"
exec python /opt/glados/app/glados_server.py --uri tcp://0.0.0.0:10200 --voice-dir /voice-glados --length-scale "$LENGTH_SCALE" $DEBUG_FLAG
