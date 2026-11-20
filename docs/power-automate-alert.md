# Weekly model-health email with Power Automate

The weekly GitHub Actions job (`.github/workflows/monitor.yml`) refreshes
`output/monitoring_report.json`. This Power Automate cloud flow reads that file
over HTTP every Monday, and if any batch is not `ok`, emails a table of the
results. It takes about 30 minutes to build.

```
Recurrence (Mon 10:00) ─► HTTP GET report ─► Parse JSON ─► Condition: overall_status ≠ "ok"
                                                              ├─ yes ─► Create HTML table (summary) ─► Send an email
                                                              └─ no  ─► (nothing)
```

## Before you start

- Sign in at <https://make.powerautomate.com> with a work or school Microsoft account.
- The **HTTP** action is a premium connector. If your account does not include
  it, sign up for the free **Power Apps Developer Plan** with the same account; it
  creates a developer environment with premium connectors for building and testing.
  Switch to that environment (top right) before creating the flow.

## Build the flow

1. **Create → Scheduled cloud flow.** Name it `Claims model health`, repeat every
   1 week on Monday, 10:00, time zone Pacific.
2. **+ New step → HTTP.**
   - Method: `GET`
   - URI: `https://raw.githubusercontent.com/Yang-Zhang1994/claims-frequency-model/main/output/monitoring_report.json`
3. **+ New step → Parse JSON.**
   - Content: the HTTP action's **Body**.
   - Schema: paste the schema below (or use *Generate from sample* with the current file).
4. **+ New step → Condition.** `overall_status` (from Parse JSON) **is not equal to** `ok`.
5. In **If yes**:
   - **Create HTML table**. From: `summary`. Columns: Automatic.
   - **Send an email (V2)** (Office 365 Outlook). To: yourself. Subject:
     `Claims model: @{body('Parse_JSON')?['overall_status']}`. Body: the HTML table's **Output**,
     plus a line with `generated_at`.
6. **Save → Test → Manually.** Because the simulated `claims_trend` batch is
   always an alert, the test run should send the email.
7. Take a screenshot of the flow designer and of the email for the README.

## Parse JSON schema

Only the fields the flow uses:

```json
{
  "type": "object",
  "properties": {
    "generated_at": { "type": "string" },
    "overall_status": { "type": "string" },
    "summary": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "batch": { "type": "string" },
          "status": { "type": "string" },
          "action": { "type": "string" },
          "actual_to_expected": { "type": "number" },
          "drifted_features": { "type": "string" }
        }
      }
    }
  }
}
```

## Ideas to extend it

- Post to a Teams channel instead of email (**Post message in a chat or channel**).
- Add a second condition that only emails on `alert`, and logs `watch` rows to an Excel table in OneDrive.
