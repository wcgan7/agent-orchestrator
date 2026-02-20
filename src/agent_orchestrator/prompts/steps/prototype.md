Build a timeboxed prototype to validate feasibility and reduce uncertainty for this task.

Mission:
- Produce a minimal proof-of-concept that answers the highest-risk technical questions.
- Prioritize learning speed and decision quality over production completeness.
- Surface concrete evidence to guide next-step decisions.

Prototype rules:
- You may implement throwaway code if it is the fastest path to validate assumptions.
- Keep scope intentionally narrow; avoid polishing, broad refactors, or production hardening.
- Do not assume infrastructure, permissions, credentials, or external services not explicitly available.
- If blocked, state exactly what is blocked and what was still learned.

Execution focus:
- Identify 1-3 hypotheses to validate.
- Run the smallest viable experiments to validate or falsify each hypothesis.
- Record observed outcomes, not just intentions.
- Keep changes targeted to the prototype objective.

Validation expectations:
- Run lightweight checks appropriate to the prototype scope (smoke tests, minimal test run, command output, or reproducible manual checks).
- If full verification is skipped, state why and what risk remains.

Output requirements:
- Return only a concise prototype summary with:
  - hypotheses tested,
  - experiments run,
  - results/evidence,
  - key limitations and risks,
  - recommendation (proceed / revise approach / stop).
- No conversational preface, no follow-up questions.
