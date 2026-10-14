# Evaluated FIGs

As of: 2026-10-06. Which Fast Information Groups (FIGs) from the FIC the radio evaluates.
The basis is ETSI EN 300 401 V2.1.1. The dispatcher for all FIGs is `parse_fib`
([src/dab/fig.py](../src/dab/fig.py)). It skips FIGs of other types and extensions;
everything it reads ends up in `dab.fig.Ensemble`.

## FIG 0 (MCI/SI)

| FIG | Content | What of it is used | What for |
|---|---|---|---|
| **0/0** | Ensemble information | only the Al flag; not the change flags and the CIF counter | alarm announcements for all services |
| **0/1** | Subchannel organisation | start address, size in CU, protection profile (UEP/EEP), bit rate | MSC decoding |
| **0/2** | Service organisation | components per service: SubChId and ASCTy (DAB/DAB+), DSCTy for stream data, SCId for packet data | playback, data services |
| **0/3** | Components in packet mode | subchannel, packet address, DSCTy per SCId | data services in packet mode |
| **0/8** | Global component definition | SCIdS → subchannel or SCId | assignment of the user applications |
| **0/10** | Date and time | time of the ensemble | time parameters of the slideshow, clock |
| **0/13** | User applications | Slideshow, SPI, TPEG, Journaline … per component | data services, X-PAD |
| **0/14** | FEC in packet mode | FEC scheme per subchannel | packet FEC |
| **0/17** | Programme type | PTy code per service; not the S/D flag | station list |
| **0/18** | Announcement support | ASu flags and clusters per service | announcements |
| **0/19** | Announcement switching | ongoing announcement: cluster, type, subchannel, New flag | announcements |
| **0/21** | Frequency information | frequencies of DAB ensembles, own and others; lists for FM, DRM and AMSS are skipped | programme guide, channel of another ensemble |
| **0/24** | Services in other ensembles (OE Services) | EIds of the ensembles per SId, for own and foreign services; the short form without EIds removes the entry; not the CAId | other channels of a station, e.g. “5C [11D, 12D?]” |
| **0/25** | Announcement support for other ensembles (OE) | ASu flags, OE clusters and EId of the targets | announcements from other ensembles |
| **0/26** | Announcement switching in other ensembles (OE) | ongoing announcement there, including the target SId | announcements from other ensembles |

Notes:

- **0/25 and 0/26** were not broadcast by any of the channels 5C, 5D, 7A and 8C on 2026-10-06.
  They are only tested with vectors according to the standard.
- **0/24** was broadcast on 2026-10-06 by 5D, 7A and 8C. 5D names its own ensemble in it,
  which is therefore dropped from the display. The database only arrives complete about every
  2 minutes; the channel list therefore stores it and adds to it during reception
  (`EnsembleInfo.service_ensembles`). Which channel belongs to an EId is worked out by
  [src/dab/service_following.py](../src/dab/service_following.py): first channels on
  which the ensemble was received here, then those that FIG 0/21 announces (marked with
  “?” in the display).

## FIG 1 (Labels)

| FIG | Content |
|---|---|
| **1/0** | Ensemble label and EId |
| **1/1** | Label of a programme service (16-bit SId) |
| **1/5** | Label of a data service (32-bit SId) |

The character flags of the short labels are not evaluated.

## Broadcast, but not yet evaluated

| FIG | Content | Observed on |
|---|---|---|
| **0/5** | Language of a subchannel | 5C |
| **0/6** | Service linking (linked services, regional variants, FM PI codes) | 5D, 7A, 8C |
| **0/7** | Configuration information: number of services, reconfiguration counter | 7A, 8C |
| **0/9** | Local time offset and ECC | all channels |

## Not observed on air

FIG 2 (extended labels), FIG 1/4 and 1/6 (component and X-PAD labels), FIG 5
(paging, TMC, old EWS), FIG 6 (conditional access) and FIG 0/20.
