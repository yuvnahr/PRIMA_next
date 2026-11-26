# Local Markdown issue tracker

## Conventions

- Keep each feature in `.scratch/<feature-slug>/`.
- Write its spec to `.scratch/<feature-slug>/spec.md`.
- Write each implementation ticket to `.scratch/<feature-slug>/issues/<NN>-<slug>.md`. Start numbering at `01`.
- Record triage state in a `Status:` line near the top of each ticket. Use the values in `triage-labels.md`.
- Append comments and conversation history under `## Comments`.

When a skill asks you to publish to the tracker, create the relevant file in the feature directory. When it asks you to fetch a ticket, open the file at the referenced path or look up the ticket number in that feature's `issues/` directory.

## Wayfinder

- Keep the map in `.scratch/<effort>/map.md` with Notes, Decisions-so-far, and Fog sections.
- Put each question in `.scratch/<effort>/issues/<NN>-<slug>.md`. Record the ticket type in a `Type:` line and its claim or resolution state in a `Status:` line.
- Record dependencies with a `Blocked by: NN, NN` line. A ticket is unblocked when every listed dependency is resolved.
- Pick the first open, unblocked, unclaimed ticket by number. Set `Status: claimed` and save before working.
- To resolve a ticket, append the answer under `## Answer`, set `Status: resolved`, and add a link to the answer in the map's Decisions-so-far section.
