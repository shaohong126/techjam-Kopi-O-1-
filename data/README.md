# Competition Data

## `public_set.jsonl`

Contains 200 labeled development sessions: 80 Buying, 80 Browsing, 30 Intent Override, and 10 Boundary sessions.

Each session contains a safe aggregate `user_profile` and public labels for local development. Direct user identifiers, timestamps, free-text reviews, raw purchase history, hidden intent cards, and simulator-policy internals are not shipped in this participant file.

## `catalog.jsonl`

Download `catalog.jsonl.gz` and the published checksum file from the organizer's official TechJam Participant Kit Release:

https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit

Verify the archive using the organizer-provided SHA-256 checksum, then decompress it and place the resulting file at `data/catalog.jsonl`.

Expected row count: 50,000. The catalog is intentionally not tracked in this team repository. See [`../DATA_ATTRIBUTION.md`](../DATA_ATTRIBUTION.md) for source attribution and use notes.

Never place API keys, private evaluation data, or participant outputs in this directory.
