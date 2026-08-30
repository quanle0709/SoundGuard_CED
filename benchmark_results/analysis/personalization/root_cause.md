# Personalization parser root cause

The failing inputs normalize punctuation before matching: `factory-method` becomes `factory method`, and `baby-blue` becomes `baby blue`. The whole-phrase regex then sees standalone `factory` and `baby`, so it correctly enforces boundaries on the *normalized* text but has already lost evidence that the token came from a hyphenated compound. `factory` activates `factory_warehouse_worker` and `factory_warehouse`; `baby` activates `parent_infant_caregiver` and the infant responsibility.

This is a general punctuation-normalization ambiguity, not a failure of the final profile validator. Of 12 diagnostic compounds, 10 produced at least one role. These expected outcomes were hand-defined as non-role technical/adjectival uses; no LLM supplied ground truth. The companion CSV contains normalized forms and exact inferred roles/contexts. No production parser change was made.
