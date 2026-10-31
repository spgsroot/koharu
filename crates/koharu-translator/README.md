# Koharu Translator

koharu-translator provides one segment-preserving translation interface for
local GGUF models and hosted providers.

Model metadata is provider-neutral: each model has a stable provider ID, a
display name, and (for local models) selectable GGUF quantizations. Every
provider module owns its connection config, defaults, request mapping, and
model catalog or discovery.

Reasoning is a switch plus an optional level. `GenerationConfig::reasoning`
turns it on or off and `reasoning_effort` picks a `ReasoningEffort` while it is
on; `None` keeps the provider's own default. A model lists the levels it
accepts in `Model::reasoning_efforts`, discovered from the provider's model
list where it reports them and otherwise taken from the provider's documented
per-model table. An empty list means the model only offers the switch. A level
the selected model does not list is rejected instead of being sent or remapped,
and a level is never sent while reasoning is off.

The crate owns only provider connectivity. Provider settings such as base URLs
are persisted under `[providers]`; credentials remain in the platform keychain.
Translation model selection, target language, instructions, and generation
options belong to the caller. `koharu-pipeline` owns those values as the
configuration of its translation processor.

`Translator` is an execution engine. A caller supplies a `ModelSelection`,
`GenerationConfig`, and `TranslationRequest` for each operation. The engine
keeps the selected local model resident when possible and reads live provider
connection settings without owning workflow configuration.
