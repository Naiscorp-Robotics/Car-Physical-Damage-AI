# Demo images

Seven photographs used by the Gradio examples and the Hugging Face model card.
They are inspection-style shots chosen because readable plates / faces were
either absent or not the focus — still re-check before any wider redistribution.

| File | Role |
|---|---|
| `01_multi_damage.jpg` | Several regions, multiple classes |
| `02_tear_scratch.jpg` | Tear + paint scratch |
| `03_side_panel.jpg` | Side panel |
| `04_crease.jpg` | Single high-confidence crease |
| `05_missed_at_default.jpg` | Failure case at threshold 0.7 |
| `input.jpg` / `input2.jpg` | Extra examples |

Do **not** add customer inspection photos here. Before committing any new image:

- readable licence plates
- identifiable faces
- phone numbers, VINs, addresses
- EXIF location (`exiftool -all= image.jpg`)
