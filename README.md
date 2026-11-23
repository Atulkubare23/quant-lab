# quant-lab

Pricing engines and quant models rebuilt from the open-source and academic projects people use most.
Dependencies are numpy and scipy only, every formula is written out, and every module has a check
against a closed form, a second method or a published table.

```
python -m pytest -q          # 75 tests, ~3-7 min (deep hedging trains twice)
python -m qlab.<module>      # each module prints a short demo
```

| Module | What it does | Built after | Checked against |
|---|---|---|---|
| `fourier` | One characteristic-function interface for Black-Scholes, Merton, Kou, Variance Gamma, NIG, CGMY, Heston and Bates. Three pricers: Carr-Madan FFT, COS and Lewis | FinancePy / QuantLib analytic engines, `fourier-option-pricer`, Fang & Oosterlee (2008) | Black-Scholes, the Merton Poisson series, agreement between the three pricers, martingale and parity checks |
| `dual`, `curves` | Dual-number forward-mode AD. Joint Newton solve of an overnight (SOFR-style) discount curve and a term forecast curve. Par-rate risk through the inverse Jacobian | rateslib (`Dual`, `Solver`), ORE / Strata curve builders | Repricing every instrument to 1e-12, recovering a flat curve, risk equal to bump-and-rebuild |
| `hullwhite` | HW1F fitted to the curve: closed-form bonds, bond options, caps, Jamshidian swaptions, a trinomial tree for Bermudans, exact Monte Carlo, σ calibration to a normal vol | QuantLib `HullWhite`, `TreeSwaptionEngine`, Brigo & Mercurio | Tree vs Jamshidian (0.3%), MC vs closed form, parity, Bermudan between max and sum of Europeans |
| `xva` | Netting-set exposure under HW with exact per-path floating fixings. Two-way CSA (threshold, MTA, MPoR). EE, ENE, PFE, then CVA, DVA, FVA with first-to-default | ORE exposure and XVA analytics | Discounted EE on reset dates = European swaption; martingale of discounted MtM; CVA(payer) = DVA(receiver) |
| `lsm` | Longstaff-Schwartz for American and Bermudan options (one or several assets); CRR tree | Longstaff & Schwartz (2001), Glasserman ch. 8 | LS table 1 (4.472 vs 4.478), CRR, Broadie-Glasserman max-call 13.90 |
| `aad` | Reverse-mode AD on numpy arrays. A basket call gives all deltas, vegas and rho in one backward pass; a 52-step Asian runs on the tape | Savine, *Modern Computational Finance* (2018); Capriotti & Giles | Bump Greeks with common random numbers to 1e-5; Black-Scholes delta and vega |
| `rough` | Rough Bergomi by the hybrid scheme (κ = 1, FFT convolution); smiles and ATM-skew term structure | McCrickerd & Pakkanen `rough_bergomi`; Bayer, Friz & Gatheral | Var(Y_t) = t^{2H}; E[V_t] = ξ₀; η = 0 gives Black-Scholes; rougher H gives a steeper skew decay |
| `mm` | Avellaneda-Stoikov inventory quotes vs a symmetric quoter; GLFT quotes with an inventory cap | Avellaneda & Stoikov (2008); Guéant, Lehalle & Fernandez-Tapia (2013) | The paper's γ = 0.1 table: 64.9 / 6.6 / 3.0 here vs 65.0 / 6.6 / 2.8 in the paper |
| `deephedge` | Deep hedging (Buehler et al. 2019): an MLP trained with Adam on the `aad` tape (no PyTorch) to minimise the entropic risk of hedged P&L, with transaction costs, under Black-Scholes or Heston | Buehler's `deephedging` repo | No costs: learns the Black-Scholes delta and price. With 0.5% costs it beats delta hedging (indifference price 3.33 vs 3.84) |
| `fmm` | SOFR forward market model (Lyashenko-Mercurio): backward-looking compounded rates, shifted lognormal, vol decaying through the accrual period, spot measure. Caplets, swaptions, Bermudans by LSM | Lyashenko & Mercurio (2019); QuantLib LMM | Numeraire reprices the curve; caplets match the σ²(T₀ + τ/3) closed form; swaptions match Rebonato; the Bermudan has the right limits |
| `cds` | ISDA Standard Model conventions: IMM coupon dates and the semi-annual roll, step-in and T+3 settlement, ACT/360. Exact protection and accrual-on-default legs on flat-forward curves. Conventional spread ↔ upfront, bootstrap, CS01 | ISDA/Markit JPMCDS model | Quote round-trips to 1e-12; legs match daily integration to 1e-6; bootstrap reprices par quotes |
| `saccr` | Basel SA-CCR: supervisory duration, delta and maturity factor; IR, FX, credit, equity and commodity add-ons; multiplier; margined and unmargined RC; EAD | BCBS 279 (2014) | **All five worked examples in Annex 4a: EAD 569, 381, 5,406, 936, 1,879** |
| `frtb`, `sbm` | The full FRTB standardised approach. Sensitivities-based method (MAR21) for all seven risk classes (GIRR, CSR non-securitisation, CSR securitisation CTP and non-CTP, equity, commodity, FX): delta, vega and curvature under three correlation scenarios. Default risk charge (MAR22): LGD, maturity scaling, seniority-aware netting, hedge benefit ratio, and the securitisation and CTP rules. Residual risk add-on (MAR23). Total = SBM + DRC + RRAO | Basel Framework MAR21–23 (2023) | GIRR correlations reproduce MAR21 Table 2; the MAR22 FAQ stock-vs-index-future offset nets to zero; hand calculations for HBR, CTP half-weighting of negative buckets and the RRAO exclusions; long options carry no curvature charge |
| `simm` | ISDA SIMM engine: IR delta (sub-curves, inflation, xccy, concentration), vega, curvature (λ/θ), equity, FX, ψ across risk classes, product classes | ISDA SIMM methodology | Algebra checked by hand calculations. **The shipped calibration is a placeholder** (see below) |

## Results worth knowing

- **Smiles, 1y, S = 100:**
  - Heston (ρ = −0.7) runs from 23.5% at the 80 strike to 14.8% at 120.
  - CGMY with Y = 0.5 is almost flat, with a slight smile.
  - Kou's jumps lift the whole smile to about 24%.
- **Multi-curve:**
  - The term index sits 23–27bp over the overnight curve.
  - A 1m receiver 7y swap at 4.00% has −607 per bp of risk to the 7y term swap and almost none elsewhere. This is what par-rate risk should look like.
- **Bermudan swaption:** a 5nc1 into 10y payer adds 0.44 points on top of the 5y × 5y European (3.12 vs 2.68 points), with a = 5% and σ = 1%.
- **xVA:**
  - For a 100m 10y payer netted against a 60m 5y receiver, a 1m-threshold CSA with a 10-day MPoR cuts CVA from 216k to 47k.
  - It cuts peak PFE from 17.0m to 3.4m.
- **AAD:** price plus all 21 Greeks of a 10-asset basket costs about 1.3 times a single pricing.
- **Rough Bergomi:**
  - With H = 0.1, the ATM-skew slope in log-log is −0.31 between 2 weeks and 8 months.
  - The asymptotic value is −0.40; the discretisation flattens it at short maturities.

## Conventions and limits

- **Curves:**
  - Times are in years, with no calendars or day-count bases.
  - Fixed legs are annual and floating legs quarterly.
  - Curves interpolate log-linear on discount factors.
- **Hull-White and xVA:** these are single-curve. The tree needs exercise and payment dates on its time grid.
- **CSA:** two-way, with no initial margin.
- **LSM:** the estimate is low-biased (it applies the exercise rule on fresh paths). There is no upper bound.
- **AAD:** uses pathwise derivatives. Digital payoffs would need smoothing first.

## SIMM parameters

`qlab/data/simm_params.json` holds **placeholder values**, so SIMM numbers from this repo are illustrative only. isda.org refused the automated download of the methodology. To make SIMM numbers real:

1. Download the current ISDA SIMM methodology PDF from isda.org.
2. Copy its risk weights, correlations, thresholds, HVR/VRW and ψ into the same JSON keys.
3. Check the curvature sign convention against the document's formula.

Every result carries the parameter file's `status` string, so placeholder output can't pass for real output.

## Not built yet

- FRTB: securitisation DRC risk weights must be supplied per tranche (the banking-book securitisation framework is not built). The internal models approach (IMA) is not built.
- SIMM credit and commodity risk classes.
- An exchange holiday calendar for the CDS dates; the engine currently skips weekends only.
