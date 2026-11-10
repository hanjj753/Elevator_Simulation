# Elevator research workflow

The user wants the research question, hypotheses, experimental methods, results,
and interpretation updated after every analysis.

- Keep `RESEARCH_STATUS.md` as the current Korean research summary. Update it
  after each completed analysis with the current hypotheses, evidence and
  limitations, and the next unresolved question.
- Append an entry to `ANALYSIS_LOG.md` for each completed analysis. Preserve
  earlier entries; distinguish exploratory discovery from a subsequent test on
  fresh data. Record the hypothesis, design, input/output paths, results and
  interpretation changes.
- Before a new follow-up simulation, save its protocol, seeds, planned sample
  size, primary outcome and contrast before generating results. A local plan
  is not external preregistration. Do not relabel a post-hoc hypothesis as an
  original hypothesis or alter a failed hypothesis to fit the result.
- Preserve historical outputs. Use a distinct output directory for a new
  experiment. Keep discovery and follow-up results separate before any pooled
  display. Five order replicates within one demand seed are not five
  independent demand observations.
- Report effects in the direction actually observed, including null or
  adverse findings. Keep P95-of-each-run averages distinct from pooled P95.
- Treat passenger counts as journey counts in simulations, not real sampled
  humans. State the scope of boarding rejection metrics. Do not infer vehicle
  routing causes or real-building performance without supporting evidence.
- Link the updated research summary and briefly explain what changed in each
  analysis handoff. Do not create recurring monitoring or automatic runs
  unless the user requests those.
