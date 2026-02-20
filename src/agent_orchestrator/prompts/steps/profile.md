Establish a performance baseline for this task.

Mission:
- Measure current performance behavior before any optimization work.
- Identify impactful bottlenecks with concrete evidence.
- Produce baseline data that the plan and benchmark steps can use for comparison.

Scope:
- Measurement and diagnosis only.
- Do NOT implement optimizations in this step.
- Do NOT choose a final optimization strategy or produce an implementation plan.

Profiling rules:
- Prefer reproducible measurements over ad-hoc observations.
- Record workload/input assumptions used for each measurement.
- Capture environment/tooling limitations that may affect fidelity.
- Distinguish measured facts from inference.
- If profiling tools are unavailable, use the best available fallback and state limitations explicitly.

Output format (use these exact section headings):

## Objective and Measurement Scope
- Restate the performance objective being evaluated.
- Define measured scope (paths, endpoints, jobs, queries, etc.).

## Baseline Measurements
- Report key baseline metrics (e.g., latency, throughput, CPU, memory, I/O).
- Include measurement method, sample size/run count, and observed range/variance.

## Hotspots and Evidence
- List top bottlenecks with concrete evidence.
- Include files/components/functions/queries implicated where possible.

## Constraints and Measurement Limits
- Note environment/tooling/data limits affecting confidence.
- State what could not be measured and why.

## Optimization Candidates (No Final Selection)
- 1-3 plausible optimization directions.
- For each: expected impact, complexity, and risk.
- Do not select a final approach in this step.

## Benchmarking Inputs
- Define what benchmark step should validate after implementation.
- Include target metrics or expected improvement bands when possible.

Quality bar:
- Return only the profiling output body.
- No conversational preface, no implementation plan, no code changes.
- Be concise, quantitative, and evidence-driven.
