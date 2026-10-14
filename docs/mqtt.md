# The radio's MQTT interface

The radio reports its state to an MQTT broker and accepts commands through it, so that
a smart-home server (Home Assistant, openHAB, ioBroker, Node-RED, Domoticz …) can read
and control it, and show notifications with buttons on its display. The code lives in `src/net/mqtt.py` (the topics),
`src/net/radio_state.py` (state and commands) and `src/ui/notifications.py`
(notifications).
The interface has not yet been tested against a real broker; what is written here
describes what the code does.

## Principles

- **The radio is the client.** It connects to the smart home's broker and opens no port
  itself. Who may log in and whether the connection is encrypted is set at the broker;
  the radio brings user name, password and TLS.
- **Everything is off until you switch it on**, on the touch page “Settings” ›
  “MQTT”. Only the touch surface (`--touch`) speaks MQTT; the desktop does not.
- **State as retained topics.** Each section of the state lies as JSON under a topic of
  its own; the broker keeps the latest state, and a newly connected client receives it
  at once. After every connection the radio sends everything again.
- **Commands via topics, answer under `result`.** The radio reports what became of a
  command, and when it rejects one, a text saying why.
- **Everything is English**, whatever language the radio's surfaces show (German can be
  chosen on the touch page “Settings” › “Display” or in the desktop's menu “View” ›
  “Language”): commands, values, the names of the entities in the smart home, answers,
  status lines, error messages, programme types. Station names, Dynamic Label texts,
  titles and artists are broadcast content and come as the station sends them. A home
  automation should still key on the machine values (JSON keys, section names, states
  such as `receiving`, SIds, `ok`, ON/OFF) rather than on texts, which may change between
  versions.

## Setup

On the touch page “Settings” › “MQTT”:

| Row | Meaning |
| --- | --- |
| MQTT | On/Off. Off ends the connection with an `offline` under `available`. |
| Connection | What the bridge is doing right now: “Connecting to …”, “Connected to …”, “The broker … refuses: …”, “No connection to …, still trying …”. Reconnects by itself, with a growing interval of up to 60 s. |
| Broker | Name or IP address, and port (default 1883; switching TLS on moves 1883 to 8883 and back). |
| Login | User name and password; leave both empty if the broker does without. The password is stored in plain text in the radio's settings (`settings.ini`). |
| TLS | Encrypted with “Verify certificate” (against the system's root certificates) or without verification (encrypted, but any certificate is accepted). |
| Discovery | Announces the radio with its entities following the Home Assistant scheme, which openHAB, Domoticz and others also read; next to it the prefix under which the smart home expects the announcements (default `homeassistant`). |
| Topic prefix | All of the radio's topics lie under this prefix (default `dabradio`). Two radios in one house get two prefixes. |

Every change takes effect immediately; the connection is then re-established. If the
prefix changes or Discovery is switched off, the radio clears the old retained topics at
the broker, so that nothing outdated is left lying there.

## Topics at a glance

All topics lie under the prefix, here `dabradio`.

| Topic | Direction | Content |
| --- | --- | --- |
| `dabradio/available` | Radio → broker | `online` or `offline` (retained). The radio's Last Will sets `offline` when the connection breaks. |
| `dabradio/state/<section>` | Radio → broker | One section of the state as JSON (retained), see [Reading](#reading). |
| `dabradio/slide` | Radio → broker | The current slideshow image, the image data itself (JPEG, PNG …, retained); empty if there is none. |
| `dabradio/station` | Radio → broker | The station playing, by the name that `set/station` accepts (retained). |
| `dabradio/output` | Radio → broker | The selected audio output, by the name that `set/output` accepts (retained). |
| `dabradio/set/<key>` | Broker → radio | A value for an entity, see [Simple values](#simple-values-set). |
| `dabradio/command/<name>` | Broker → radio | A command with arguments as a JSON object, see [Commands](#commands-command). |
| `dabradio/result` | Radio → broker | What became of a command (JSON, not retained). |
| `dabradio/notification/answer` | Radio → broker | How a notification ended: the button pressed, or why not (JSON, not retained), see [Notifications](#notifications). |
| `homeassistant/<component>/<node>/<key>/config` | Radio → broker | The discovery announcements (retained), see [Discovery](#discovery). |

The radio subscribes to `dabradio/command/#` and `dabradio/set/#` with QoS 1 and
publishes the answers of notifications with QoS 1, so that neither a command nor an
answer is lost on the way; everything else goes with QoS 0.

## Reading

### Sections

Each section is a JSON object under `dabradio/state/<section>` and is published again as
soon as its content changes. The sections `signal` and `playback` change several times a
second and therefore go out at most once per second, the first change at once (for
`playback` only when the position alone moves; pause and stop come at once).

Service ids (SIds) are written as four hex digits, as the radio shows them (`"D220"`).
Whatever does not exist or is not known is `null`.

#### `receiver`: the receiver

```json
{
  "state": "receiving",
  "status": "RTL-SDR ready @ 178.352 MHz, 2.048 MSps",
  "error": null,
  "device": "RTL-SDR",
  "channel": "5C",
  "frequency_mhz": 178.352,
  "ensemble": "DR Deutschland",
  "ensemble_id": "10BC",
  "scan": null,
  "version": "1.0.0"
}
```

| Field | Meaning |
| --- | --- |
| `state` | `idle` (no reception), `connecting` (the device is being opened), `receiving`, `scanning` (scan). |
| `status` | The radio's status line, in English. |
| `error` | The last device error, until reception or a scan is next started. |
| `device` | The selected receiver: “RTL-SDR”, “RSPduo Tuner 1”, “RSPduo Tuner 2”. |
| `channel` | The Band III channel (`"5C"`), `null` with a manual frequency. |
| `ensemble`, `ensemble_id` | Name and EId of the ensemble of the station list on display. |
| `scan` | During a scan `{"channel": "5A", "index": 0, "total": 38}`. |

#### `services`: the station list

```json
{
  "channel": "5C",
  "live": true,
  "complete": true,
  "list": [
    {"sid": "D210", "name": "Dlf", "programme_type": "News", "favourite": true},
    {"sid": "D220", "name": "Dlf Kultur", "programme_type": null, "favourite": false}
  ]
}
```

`live` says whether the list is being received or is stored, `complete` whether the
received one is complete. While it is still incomplete, this holds, as on the surfaces,
the stored list of the channel, so that the names are not missing for seconds after
switching.

#### `service`: the selected station

```json
{
  "sid": "D220",
  "name": "Dlf Kultur",
  "favourite": false,
  "programme_type": "Culture",
  "codec": "DAB+",
  "bitrate_kbps": 96,
  "audio_format": "48 kHz, Stereo, SBR"
}
```

`codec` is `"DAB+"` or `"DAB"`, `audio_format` the description of the audio as the
user interface shows it; both `null` until the audio is playing.

#### `now_playing`: Dynamic Label and title

```json
{"text": "a-ha - Take On Me", "title": "Take On Me", "artist": "a-ha", "item_running": true}
```

`text` is the Dynamic Label. `title` and `artist` come from DL Plus and are only present
while the station reports a piece of music playing (`item_running`); stations without
DL Plus deliver only `text`, and `item_running` is then `null`.

#### `audio`: audio

```json
{
  "muted": false,
  "volume_percent": 35,
  "volume_db": -39.0,
  "volume_device_percent": null,
  "volume_ceiling_db": 0,
  "output": {"id": "alsa_output.platform-soc_sound.stereo-fallback", "name": "Built-in Audio", "bluetooth": false, "default": true},
  "output_choice": "",
  "outputs": [{"id": "…", "name": "…", "bluetooth": false, "default": true}],
  "equalizer": {"enabled": false, "per_station": false, "station_own": false}
}
```

`volume_percent` is the slider position, `volume_db` the gain it gives (`null` at
0 %), `volume_ceiling_db` the “Maximum volume”. On a Bluetooth device whose own volume
the system sets (only on the touch radio started through `pi/start-touch.sh`), the
slider is linear over the device's range, as a phone's:
`volume_device_percent` is that share in percent, and `volume_db` the nominal value
of it on PipeWire's cubic scale (what the device makes of it is its own curve); for
every other output `volume_device_percent` is `null`. `output` is the output the audio is
currently playing on, `output_choice` the selected one: `""` for “Automatic” (the
system's default output, i.e. headphones as soon as they are connected; with several
Bluetooth devices connected to the touch radio on the Pi, the one connected last),
otherwise the `id` from `outputs`.

#### `signal`: reception quality (at most once per second)

```json
{
  "quality": {"bars": 4, "text": "very good"},
  "frames": {"unit": "Superframes", "synced": 1250, "total": 1250, "lost": 0, "rs_corrections": 3},
  "frequency_offset_hz": -118.0,
  "clipping": 0.0,
  "sfn_others": 0.12,
  "sfn_outside_guard": 0.0,
  "adc_overload": false,
  "load": {"dsp": 0.41, "dsp_waiting": 0.02, "cpu": 0.8}
}
```

`quality` is the reception indicator of the radio page: `bars` 0 to 4 (“no audio”,
“poor”, “fair”, “good”, “very good”), `null` without audio frames. `frames` counts the
audio frames of the selected station since it was selected (DAB+: “Superframes” with
Reed-Solomon corrections, DAB: “MP2 frames”). `clipping` is the share of clipped
samples (0 to 1), `sfn_*` the share of other paths of the single-frequency network and,
of those, outside the guard interval, `load` the processing load (shares of real time,
`cpu` in cores).

#### `announcement`: announcements

```json
{
  "active": {"types": ["Road Traffic flash"], "alarm": false, "new": true, "source": "Dlf Kultur"},
  "elsewhere": null,
  "visit": null,
  "switching": true
}
```

`active` is an announcement running for the selected station in its own ensemble
(`source`: the station whose audio carries it), `elsewhere` one in another ensemble
(with `ensemble_id`), `visit` the visit there when “Switch to announcements”
(`switching`) is on: channel, SId and name of the station there, plus
`origin_channel`, `origin_sid`, `origin_name` of the station left. `alarm` marks an
alarm announcement; `types` names the kinds (“Road Traffic flash”, “Area weather flash”,
“Alarm” …).

#### `favourites`: the favourites

```json
{"list": [{"channel": "5C", "sid": "D210", "name": "Dlf", "ensemble": "DR Deutschland"}]}
```

Across all channels, in the order of Band III. A favourite on a channel that was never
stored has its SId as `name`.

#### `recorder` and `recordings`: recording

```json
{
  "enabled": true,
  "directory": "/media/dabradio/STICK/Radio",
  "available": true,
  "recording": true,
  "message": "Now recording",
  "current": {"id": "…", "name": "a-ha – Take On Me", "station": "Dlf Kultur", "…": "…"},
  "ring_size": 10
}
```

`recordings` has two lists: `ring` (the recent titles, oldest first) and `kept`
(the kept ones). Each recording:

| Field | Meaning |
| --- | --- |
| `id` | Identifier for `keep_recording`, `delete_recording`, `play_recording`. |
| `name` | “Artist – Title”, one of the two, the Dynamic Label or “Untitled”. |
| `station`, `ensemble`, `channel`, `sid` | Where the title came from. |
| `title`, `artist`, `text` | From DL Plus, or the Dynamic Label last seen. |
| `start`, `duration` | Start in Unix seconds, duration in seconds. |
| `size`, `lost` | Bytes of the file, lost audio units. |
| `codec`, `bitrate_kbps` | “DAB+” or “DAB”, bit rate. |
| `kept`, `complete` | Kept; complete (the recording in progress is not). |

#### `playback`: playing a recording

```json
{"id": "…", "name": "a-ha – Take On Me", "state": "playing", "position": 12.5, "duration": 221.0}
```

`state` is `loading`, `playing` or `paused`; without playback all fields are `null`.
The position comes at most once per second.

#### `notifications`: notifications open on the display

```json
{
  "shown": "door",
  "open": [
    {
      "id": "door",
      "title": "Front door",
      "message": "Someone rang the bell.",
      "level": "info",
      "buttons": [{"id": "open", "label": "Open", "style": "accent"}, {"id": "2", "label": "Ignore", "style": ""}],
      "image": true,
      "received": 1791000000.0,
      "expires": 1791000060.0
    }
  ]
}
```

`shown` is the id of the notification on display, `open` lists every open one in the
order the radio shows them, the one on display first. `image` says whether it brought
one (the image itself is not sent back), `received` and `expires` are Unix seconds,
`expires` is `null` without a timeout. See [Notifications](#notifications).

### The slideshow

`dabradio/slide` carries the image data of the current image as the station sends it
(usually JPEG or PNG), retained; if the station changes or there is no image, the topic
is empty. The radio reports the image type in the image's discovery config
(`content_type`); whoever reads without Discovery recognises it by the first bytes.

## Controlling

### Simple values: `set`

For the entities of a smart home, one topic with a simple value each is enough:

| Topic | Value | Effect |
| --- | --- | --- |
| `dabradio/set/station` | A name from `dabradio/station` or from the options of the station select | Plays the station: tunes to its channel, selects it and starts reception if it is not running. |
| `dabradio/set/volume` | `0` to `100` | Volume in percent of the slider position. |
| `dabradio/set/muted` | ON or OFF | Mute. |
| `dabradio/set/reception` | ON or OFF | Start or stop reception. |
| `dabradio/set/scan` | ON or OFF | Start or cancel a scan (only while reception is not running). |
| `dabradio/set/recording` | ON or OFF | Recording of the recent titles on or off. |
| `dabradio/set/announcement_switching` | ON or OFF | Switch to announcements. |
| `dabradio/set/equalizer` | ON or OFF | Equalizer on or off. |
| `dabradio/set/output` | A name from `dabradio/output` or from the options of the output select, “Automatic” for the default output | Audio output. |
| `dabradio/set/notify` | Any text | Shows the text as a notification with “OK” (as `notify` with `message` alone). |

The radio also understands ON and OFF as `on`/`off`, `true`/`false`, `1`/`0`, and the
German `an`/`ein` and `aus`, whatever the language, case-insensitive.

The station names are those of the favourites and of the stations of the ensemble
currently on display; if a name occurs twice, it carries its channel (“Dlf (5C)”).
Which names currently apply is listed in the options of the discovery config
`select/…/station` and follows from `favourites` and `services`.

### Commands: `command`

Under `dabradio/command/<name>` the radio accepts every command of the state model; the
payload is a JSON object with the arguments, or empty. A station is named by `sid` (hex
text such as `"D220"`, also `"0xD220"`, or as a number) or by `name` (as in the station
list, case-insensitive).

| Command | Arguments | Effect |
| --- | --- | --- |
| `select_channel` | `channel` | Select a channel (`"5C"`); during reception the radio retunes. |
| `select_service` | `sid` or `name` | Select a station from the list on display, without starting reception. |
| `play` | `sid` or `name`, optional `channel` | Play a station: channel, selection and start in one. Without `channel` the list on display applies; with `channel` the stored list of that channel, and an unknown `sid` is then tried anyway. |
| `start_reception`, `stop_reception` | – | Start or stop reception. |
| `start_scan`, `cancel_scan` | – | Start or cancel a scan. |
| `set_volume` | `percent` (0 to 100) | Volume. |
| `set_muted` | `muted` (bool) | Mute. |
| `set_output` | `id` (from `audio.outputs`, `""` for Automatic) | Audio output. |
| `set_equalizer_enabled` | `enabled` (bool) | Equalizer. |
| `set_favourite` | `sid` or `name`, `favourite` (bool), optional `channel` | Add or remove a favourite; without `channel` the channel on display. |
| `set_announcement_switching` | `enabled` (bool) | Switch to announcements. |
| `return_from_announcement` | – | Return from the announcement of another ensemble. |
| `set_recording_enabled` | `enabled` (bool) | Recording on or off. |
| `keep_recording` | `id` | Keep a title of the ring (the one being recorded too). |
| `delete_recording` | `id` | Delete a title of the ring or a kept one. |
| `play_recording` | `id`, optional `position` (seconds) | Play a recording instead of the live audio. |
| `pause_playback`, `resume_playback`, `stop_playback` | – | Pause, resume, stop playback. |
| `seek_playback` | `position` (seconds) | Jump within the playback. |
| `notify` | `title` and/or `message`, optional `id`, `buttons`, `level`, `timeout`, `image` | Show a notification on the display, see [Notifications](#notifications). |
| `dismiss_notification` | `id` | Close an open notification; it is answered `dismissed`. |

The radio takes Boolean arguments as JSON `true`/`false`, as `0`/`1` or as text
(`"on"`, `"off"`, `"ja"`, `"nein"`, `"an"`, `"aus"` …; the German words are accepted
whatever the language).

The radio checks every command against its state: during a scan no station is
selected, a scan does not start while reception is running, an unknown station or an
unknown recording is rejected, and so is a value outside its range.

### Answers: `result`

For every command, whether via `set` or `command`, the radio publishes under
`dabradio/result` (not retained):

```json
{"command": "play", "arguments": {"channel": "5C", "sid": "D220"}, "ok": true, "message": null}
```

On rejection `ok` is `false` and `message` says why, in English:

```json
{"command": "set/station", "arguments": {"value": "WDR 2"}, "ok": false, "message": "Unknown station “WDR 2”."}
{"command": "set_volume", "arguments": {"percent": 101}, "ok": false, "message": "“percent” must be between 0 and 100."}
{"command": "start_scan", "arguments": {}, "ok": false, "message": "Reception is running; stop it before scanning."}
```

For `set/<key>`, `command` holds the key if the value itself did not fit, otherwise the
command built from it with its arguments. Whether an accepted command also takes effect
is shown by the state: `play` is accepted as soon as the radio retunes; whether the
station can then actually be heard is shown in `service` and `signal`.

## Notifications

The radio shows notifications from the smart home on its display, over everything else,
and reports which button was pressed: for the doorbell (“Open” / “Ignore”), the washing
machine, a window left open when rain comes. A notification is sent with
`dabradio/command/notify`:

```json
{
  "id": "door",
  "title": "Front door",
  "message": "Someone rang the bell.",
  "buttons": [
    {"id": "open", "label": "Open", "style": "accent"},
    {"id": "ignore", "label": "Ignore"}
  ],
  "level": "info",
  "timeout": 60,
  "image": "/9j/4AAQSkZJRgABAQ…"
}
```

| Argument | Meaning |
| --- | --- |
| `id` | Chosen by the sender, at most 64 characters; it comes back in the answer. A notification with the id of an open one takes its place (the old one is answered `replaced`), and `dismiss_notification` closes it by its id. Without `id` the radio chooses one; it is in the answer and in `state/notifications`. |
| `title`, `message` | At most 120 and 1000 characters; at least one of them. Plain text: line breaks are kept, HTML is not interpreted. |
| `buttons` | Up to 4. Each one an object with `label` (at most 40 characters), optionally `id` (what the answer names; without one its position, `"1"`, `"2"` …) and `style`: `"accent"` (the colour of the radio's main buttons) or `"danger"` (red); or just the label as text. Without buttons the radio shows “OK”, which answers with the button `ok`. |
| `level` | `info` (default), `warning` or `alarm`: the colour of the bar above the title (blue, amber, red). An alarm is shown before any other; a warning before the infos. |
| `timeout` | Seconds from arrival after which the notification closes by itself (answered `timeout`), at most 86400; without it, or with `0`, it stays until it is answered. With a timeout the bar above the title shrinks from the right as the time runs out. |
| `image` | A JPEG, PNG, GIF or BMP as base64, also as a data URL (`data:image/jpeg;base64,…`), at most 4 MB: a snapshot of the doorbell camera, for instance. It stands to the left of the text. |

Whether the radio accepted the notification is under `dabradio/result` as for every
command (the image only by its length). A tap beside the dialog does nothing: it closes
only through its buttons, its timeout or `dismiss_notification`. Its buttons accept a tap
only 0.6 s after it appeared, so that a finger on its way to something else does not
press one by accident. A notification wakes the display when it is dimmed or off.

Several notifications at once are shown one after another: alarms first, then warnings,
then infos, each in the order they came; the dialog says how many more wait. An alarm
arriving while another notification is shown takes its place at once; the other waits
again. At most 20 are open at once.

### The answer

Every notification ends with exactly one answer under `dabradio/notification/answer`
(QoS 1, not retained):

```json
{"id": "door", "reason": "pressed", "button": "open", "label": "Open"}
```

| `reason` | Meaning |
| --- | --- |
| `pressed` | A button was pressed; `button` is its `id`, `label` its label. |
| `timeout` | The timeout ran out. |
| `dismissed` | Closed by `dismiss_notification`. |
| `replaced` | A notification with the same `id` took its place. |
| `closed` | The radio quit. |

Except for `pressed`, `button` and `label` are `null`. An automation should key on `id`
and `button`, not on the label, which the sender may reword.

Notifications are not retained: what was sent while the radio was offline is lost, and
an old doorbell does not reappear after a restart. Whether the radio is reachable is shown
by `dabradio/available`.

## Discovery

With Discovery the radio announces its entities following the Home Assistant scheme
(“MQTT Discovery”), which openHAB (MQTT binding, “HomeAssistant MQTT Components”),
Domoticz and others also read: one retained config per entity under
`<discovery-prefix>/<component>/<node>/<key>/config`. The node is `dabradio_<hostname>`
of the Pi, so that two radios in one house are two devices; all entities belong to the
device “DAB radio” with the radio's version as `sw_version` and `dabradio/available` as
availability. The entities read their states with templates from the sections above;
the commands go via the `set` topics.

| Component | Key | Name | State from | Command to |
| --- | --- | --- | --- | --- |
| select | `station` | Station | `dabradio/station` | `set/station` |
| select | `output` | Output | `dabradio/output` | `set/output` |
| number | `volume` | Volume | `audio.volume_percent` | `set/volume` |
| switch | `muted` | Mute | `audio.muted` | `set/muted` |
| switch | `reception` | Reception | `receiver.state` (`connecting`, `receiving`) | `set/reception` |
| switch | `scan` | Scan | `receiver.state` (`scanning`) | `set/scan` |
| switch | `recording` | Recording | `recorder.enabled` | `set/recording` |
| switch | `announcement_switching` | Switch to announcements | `announcement.switching` | `set/announcement_switching` |
| switch | `equalizer` | Equalizer | `audio.equalizer.enabled` | `set/equalizer` |
| sensor | `service` | Station | `service.name` | – |
| sensor | `ensemble` | Ensemble | `receiver.ensemble` | – |
| sensor | `channel` | Channel | `receiver.channel` | – |
| sensor | `text` | Dynamic Label | `now_playing.text` | – |
| sensor | `title` | Title | `now_playing.title` | – |
| sensor | `artist` | Artist | `now_playing.artist` | – |
| sensor | `quality` | Reception | `signal.quality.text` | – |
| sensor | `status` | Status | `receiver.status` | – |
| sensor | `recorder` | Recorder | `recorder.message` | – |
| sensor | `codec` | Codec | `service.codec` | – |
| sensor | `bitrate` | Bit rate (kbit/s) | `service.bitrate_kbps` | – |
| image | `slide` | Slideshow | `dabradio/slide` | – |
| notify | `notify` | Notification | – | `set/notify` |
| event | `notification_answer` | Notification answer | `dabradio/notification/answer` | – |

The names are English, whatever language the radio shows. Home Assistant derives the
entity ids from them when it first sees an entity (for example
`sensor.dab_radio_title`) and keeps them afterwards, since the `unique_id`
(`<node>_<key>`) does not change.

In Home Assistant the action `notify.send_message` on the notify entity shows its
`message` with “OK” (Home Assistant hands the radio no title). The event entity fires on
every answer: its `event_type` is the `reason`, and `id`, `button` and `label` are its
attributes, so an automation can trigger on it without a topic of its own. Notifications
with buttons or an image are sent with `mqtt.publish` to `dabradio/command/notify`.

The radio announces the options of the station select (favourites and stations of the
ensemble) and of the output select again as soon as they change; the image's config
again when the image type changes (JPEG, PNG), because Home Assistant takes it from the
config. What Discovery does not map (setting favourites, keeping recordings, playback)
goes via `command`.

If Discovery is switched off or one of the prefixes changes, the radio publishes empty
messages to the old config topics; the entities then disappear from the smart home.

## Examples

Listen to what the radio reports:

```sh
mosquitto_sub -h broker -u radio -P secret -t 'dabradio/#' -v
```

Select a station, set the volume, mute:

```sh
mosquitto_pub -h broker -t dabradio/set/station -m 'Dlf Kultur'
mosquitto_pub -h broker -t dabradio/set/volume -m 35
mosquitto_pub -h broker -t dabradio/set/muted -m ON
```

The same and more via commands:

```sh
mosquitto_pub -h broker -t dabradio/command/play -m '{"channel": "5C", "sid": "D220"}'
mosquitto_pub -h broker -t dabradio/command/set_favourite -m '{"name": "Dlf", "favourite": true}'
mosquitto_pub -h broker -t dabradio/command/keep_recording -m '{"id": "2026-10-10 18.32 Dlf Kultur"}'
mosquitto_pub -h broker -t dabradio/command/stop_reception -m ''
```

The slideshow image into a file:

```sh
mosquitto_sub -h broker -t dabradio/slide -C 1 > slide.jpg
```

A notification with two buttons and the answer to it:

```sh
mosquitto_pub -h broker -q 1 -t dabradio/command/notify -m '{"id": "door", "title": "Front door", "message": "Someone rang the bell.", "buttons": [{"id": "open", "label": "Open", "style": "accent"}, {"id": "ignore", "label": "Ignore"}], "timeout": 60}'
mosquitto_sub -h broker -q 1 -t dabradio/notification/answer -C 1
```

With a snapshot of a camera:

```sh
printf '{"id": "door", "title": "Front door", "buttons": ["Open", "Ignore"], "image": "%s"}' \
  "$(base64 -w0 snapshot.jpg)" | mosquitto_pub -h broker -q 1 -t dabradio/command/notify -s
```

In Home Assistant, an automation that opens the door:

```yaml
triggers:
  - trigger: state
    entity_id: event.dab_radio_notification_answer
conditions:
  - condition: template
    value_template: >
      {{ trigger.to_state.attributes.id == 'door'
         and trigger.to_state.attributes.button == 'open' }}
actions:
  - action: lock.unlock
    target:
      entity_id: lock.front_door
```

## What MQTT cannot do

Device and gain, location, equalizer curves, the recording settings beyond on/off,
Wi-Fi, Bluetooth and drives remain reserved for the user interface. A notification plays
no sound and does not lower the radio. The data services
(programme guide, GNSS, traffic) are not part of the state; nor are signal displays
such as spectrum and constellation.
