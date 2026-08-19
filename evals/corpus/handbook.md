# Northwind Robotics — Employee Handbook

Version 4.2 · Effective 2026-01-01 · Owner: People Operations

This handbook is the demo corpus for AgenticRAG. It is fiction. Every fact in it
is deliberately specific so that eval cases can be graded against it without
ambiguity — a question with two defensible answers cannot distinguish a good
retrieval system from a lucky one.

---

## 1. Leave and time off

### 1.1 Annual leave

Full-time employees accrue **26 days** of annual leave per year, accruing at
2.167 days per calendar month. Leave accrues from the first day of employment,
but may not be taken during the first **60 days**.

Up to **10 days** may be carried into the following year. Carried days expire on
**31 March**. Days beyond the carry-over limit are forfeited and are not paid out,
except on termination, where all accrued and untaken leave is paid at the
employee's base daily rate.

Part-time employees accrue pro rata based on contracted hours.

### 1.2 Sick leave

Employees receive **12 days** of paid sick leave per year. Sick leave does not
carry over. A medical certificate is required for any absence exceeding **3
consecutive working days**.

Sick leave taken during annual leave is reclassified as sick leave on production
of a medical certificate, and the annual leave days are restored.

### 1.3 Parental leave

Primary caregivers receive **18 weeks** of fully paid parental leave. Secondary
caregivers receive **6 weeks**. Both are available from the first day of
employment — there is no qualifying period.

Parental leave may be taken in up to **3 separate blocks** within 24 months of
the birth or placement.

### 1.4 Bereavement and compassionate leave

**5 days** paid leave for an immediate family member; **2 days** for extended
family. Additional unpaid leave may be approved by a director.

### 1.5 Unpaid sabbatical

After **4 years** of continuous service, employees may apply for an unpaid
sabbatical of between 1 and 6 months. Applications require 90 days' notice and
director approval. Health insurance continues during sabbatical; equity vesting
is paused.

---

## 2. Remote and hybrid work

### 2.1 Standard arrangement

Northwind operates a hybrid model. Employees based within 50km of an office
attend the office a minimum of **2 days per week**. Tuesday is the designated
all-hands anchor day; the second day is chosen by the team.

### 2.2 Fully remote

Fully remote status requires director approval and is reviewed annually. Fully
remote employees must be in a timezone within **5 hours** of their team's primary
timezone.

### 2.3 Working from another country

Employees may work from another country for up to **30 days** per calendar year
without prior approval, subject to notifying People Operations at least **14
days** in advance. Beyond 30 days, tax and immigration review is required and
approval takes approximately 6 weeks.

Work from a country under sanctions is prohibited without exception.

### 2.4 Home office allowance

A one-time allowance of **$1,500** is provided on joining, and **$500** every two
years thereafter. Receipts are required for claims above $100. Equipment
purchased with the allowance remains the property of Northwind and must be
returned on termination if its purchase value exceeded $500.

---

## 3. Expenses and travel

### 3.1 Approval thresholds

| Amount | Approver |
| --- | --- |
| Under $250 | No pre-approval; submit receipt |
| $250 – $2,000 | Line manager |
| $2,000 – $10,000 | Director |
| Over $10,000 | VP Finance and CFO |

Expenses must be submitted within **45 days** of being incurred. Claims
submitted after 45 days require a written exception from the VP Finance and are
approved only where the delay was outside the employee's control.

### 3.2 Travel

Economy class for flights under **6 hours**; premium economy for flights of 6 to
10 hours; business class above 10 hours or where the employee works on the day of
arrival. Directors and above may book business class on any flight over 6 hours.

Hotel caps: **$280** per night in New York, London, San Francisco, Tokyo and
Singapore; **$180** per night elsewhere. Caps exclude tax.

Per diem for meals is **$75** per day in capped cities and **$55** elsewhere. Per
diem is not payable where meals are provided by a conference or client.

### 3.3 Reimbursement timing

Approved expenses are reimbursed in the payroll run following approval. Payroll
runs on the **25th** of each month, or the preceding business day where the 25th
falls on a weekend or public holiday.

---

## 4. Compensation and review

### 4.1 Review cycle

Performance and compensation are reviewed twice yearly, in **April** and
**October**. Promotions may be made in either cycle. Compensation changes take
effect on the first day of the following month.

### 4.2 Equity

New joiners receive options vesting over **4 years** with a **1-year cliff**,
vesting monthly thereafter. The post-termination exercise window is **7 years**
from the grant date for employees with at least 2 years of service, and 90 days
otherwise.

### 4.3 Referral bonus

**$4,000** for a successful engineering referral and **$2,500** for all other
roles, paid after the referred employee completes **6 months** of service. The
referrer must still be employed at the time of payment.

---

## 5. Security and data handling

### 5.1 Device policy

All laptops must have full-disk encryption enabled and screen lock after **5
minutes** of inactivity. Personal devices may access email and chat via the
managed application only; they may not access source code or customer data.

Lost or stolen devices must be reported to security@northwind.example within
**2 hours** of discovery, at any hour.

### 5.2 Data classification

| Class | Examples | Storage |
| --- | --- | --- |
| Public | Marketing site, published docs | Anywhere |
| Internal | Roadmaps, org charts | Company systems only |
| Confidential | Customer data, contracts | Encrypted, access-logged |
| Restricted | Credentials, keys, PII at scale | Vault only, break-glass access |

Customer data may never be copied to a personal device, a personal cloud account,
or a third-party AI service that is not on the approved vendor list.

### 5.3 Access reviews

Access to Confidential and Restricted systems is reviewed **quarterly**. Access
not used in **90 days** is revoked automatically. Reinstatement requires the same
approval as the original grant.

### 5.4 Incident response

Suspected security incidents are reported to the security channel immediately.
The on-call security engineer acknowledges within **15 minutes** during business
hours and **60 minutes** outside them. A written post-incident review is required
within **5 business days** for any incident rated Sev-1 or Sev-2.

---

## 6. Engineering practices

### 6.1 Code review

Every change requires **1 approving review**; changes touching authentication,
billing or data deletion require **2**, at least one from the owning team.

Pull requests should be under **400 lines** of diff. Larger changes require a
written explanation in the description of why they could not be split.

### 6.2 Deployment

Deploys to production are automatic on merge to `main` once CI passes. Deploys
are frozen from **Friday 16:00** until Monday 09:00 in the deployer's local time,
and for the last **two weeks of December**. Freezes may be overridden by an
incident commander during an active incident.

### 6.3 On-call

Engineers on the on-call rotation carry the pager for **7 days**, from Wednesday
10:00 to the following Wednesday 10:00. On-call weeks are compensated at **$800**
per week, plus **$150** per night with a page between 22:00 and 07:00.

Nobody may be on-call for more than **2 weeks** in any 8-week period.

### 6.4 Incident severity

| Severity | Definition | Response |
| --- | --- | --- |
| Sev-1 | Complete outage or data loss | Page immediately, all hands |
| Sev-2 | Major feature unavailable, no workaround | Page immediately |
| Sev-3 | Degraded performance or workaround exists | Next business day |
| Sev-4 | Cosmetic or single-customer | Backlog |

---

## 7. Procurement and vendors

### 7.1 New vendors

Any new vendor processing customer data requires a security review before
signature. Reviews take approximately **10 business days**. A DPA is mandatory
for any vendor handling personal data.

### 7.2 Software purchases

Individual software subscriptions under **$50** per month may be expensed
directly. Above that, procurement must check for an existing company licence
first — duplicate licences are the single largest source of software waste.

### 7.3 Contract renewals

Renewals are reviewed **60 days** before the renewal date. Auto-renewing
contracts above $25,000 per year require a documented decision to renew.

---

## 8. Conduct

### 8.1 Conflicts of interest

Outside employment, board seats and investments in competitors or vendors must be
declared to People Operations within **14 days** of arising. Investments in
publicly traded companies below **1%** of the employee's portfolio need not be
declared.

### 8.2 Gifts

Gifts valued above **$100** may not be accepted. Meals and hospitality in the
normal course of business are exempt up to **$200** per occasion.

### 8.3 Raising concerns

Concerns may be raised with a manager, People Operations, or anonymously through
the ethics line. Retaliation for a good-faith report is grounds for termination.
Reports are acknowledged within **3 business days**.

---

## 9. Leaving

### 9.1 Notice

Notice is **4 weeks** for individual contributors and **8 weeks** for directors
and above. Notice may be waived by mutual agreement.

### 9.2 Final pay

Final pay, including accrued and untaken annual leave, is paid in the payroll run
following the last working day. Any outstanding equipment above $500 in purchase
value must be returned within **10 business days** of the last working day.

### 9.3 References

Northwind provides factual references confirming dates of employment and job
title only. Managers may not provide personal references on company letterhead.
