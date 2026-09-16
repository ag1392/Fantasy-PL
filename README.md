# Fantasy-PL: Free Hit tool

Expected points for every Premier League player, priced from **live betting
exchange odds** wherever the market can reach, and from statistical models
anchored to exchange data where it cannot. Then an exact solve of the
squad-selection knapsack.

The premise: for anything the market prices with real money, the market is a
better estimator than a model fitted on a few hundred matches. The work is in
(a) getting FPL's scoring events onto markets that are actually liquid, and
(b) being honest about the parts where no market exists.

---

## The tier system

Every projected number carries a tag saying where it came from.

| Tier | Source | Covers |
|------|--------|--------|
| **1** | Liquid exchange markets, read directly (back/lay midpoint) | Team goals, clean sheets, goals conceded, save volume |
| **2** | Thin exchange market, **rescaled to agree with Tier 1** | Player goals from anytime-goalscorer books |
| **3** | Statistical model with an **exchange-derived parameter** | Defensive contribution, cards, assists, bonus |

### Tier 1: one model, many markets

Rather than read each FPL quantity off its own book, a single Dixon-Coles
bivariate goal model is fitted to the *most liquid* markets on a fixture:
match odds, over/under lines, both-teams-to-score. Every team-level quantity
is then read off the resulting joint scoreline distribution:

```
clean sheet          P(opponent goals = 0)
goals-conceded       E[floor(goals against / 2)]
team attack          lambda, which anchors player goal rates
save volume          a fitted function of opponent lambda
```

The payoff is **mutual consistency**: clean sheets, match result and total
goals cannot disagree with one another, which they can when each is read off
a separate book.

When enough markets are available the fit is over-determined and its residual
doubles as a liquidity alarm. On the current feed there are four observations
(1X2 plus one over/under line) against three parameters, so that alarm is
weak for now.

### Tier 2: thin markets, liquid scale

*Requires Paid API tier*

Anytime-goalscorer books exist on the exchange but are shallow. They are
still useful for *relative* shape within a team, so they are used for shape
and then rescaled so the team's player goal expectations sum to the team
total implied by the liquid markets. 

### Tier 3: where no market exists

**Defensive contribution** is the interesting case: it is worth 2 points, it
is new, and no exchange market prices it. But how much defending a side does
depends on how much it is without the ball, which the match odds *do* tell
you. So the per-90 rate is scaled by the opponent's expected goals relative
to the league average. The count is negative binomial, with dispersion
solved per position so the modelled threshold hit-rate reproduces a full
season's observed rate.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Grab a free key from [The Odds API](https://the-odds-api.com/), just an
email, no betting account needed, and put it in `.env` as `ODDS_API_KEY`.
Then run:

```python
from fplfh.pipeline import run
from fplfh.optimise import optimise_free_hit

res = run()
print(optimise_free_hit(res.players, budget=100.0).summary())
```

Or use the notebooks: `notebooks/run-weekly/` to pick a squad week to week,
`notebooks/backtest/` to check the model against last season. No API key is
fine too, it just falls back to the statistical model and says so in the
output.
