# Treasury Option Underlying Futures Mapping

This note documents how the `spot` field should be interpreted for Treasury option
surface generation in this repository.

## Meaning of `spot`

In the minute-SVI generation pipeline, the output precalibration CSV stores a field
named `spot`. For Treasury options, this is a legacy field name.

It should be interpreted as:

- the relevant **underlying Treasury futures price** used to normalize strike and
  invert implied volatility

It is **not**:

- a cash bond spot price
- a generic front-quarter futures price selected only from the trade date

## CME Mapping Rule

For Treasury options on futures, the option can expire in a monthly or serial month,
while the underlying futures contract still trades on the quarterly cycle.

The CME rule is:

- Quarterly option expiries in the March cycle (`Mar`, `Jun`, `Sep`, `Dec`) map to
  the **same-month** futures contract.
- Serial option expiries outside the March cycle (`Jan`, `Feb`, `Apr`, `May`, `Jul`,
  `Aug`, `Oct`, `Nov`) map to the **next quarterly** futures contract.

For practical TY surface generation, the price input should therefore be selected from
the underlying quarterly futures contract implied by the **option expiry month**, not
by the file month or by a generic front-quarter convention.

## TY Mapping Table

For TY option contract month codes, the underlying futures month should be selected as
follows:

| Option Month Code | Expiry Month | Underlying Futures Month Code |
| --- | --- | --- |
| `A` / `M` | Jan | `H` |
| `B` / `N` | Feb | `H` |
| `C` / `O` | Mar | `H` |
| `D` / `P` | Apr | `M` |
| `E` / `Q` | May | `M` |
| `F` / `R` | Jun | `M` |
| `G` / `S` | Jul | `U` |
| `H` / `T` | Aug | `U` |
| `I` / `U` | Sep | `U` |
| `J` / `V` | Oct | `Z` |
| `K` / `W` | Nov | `Z` |
| `L` / `X` | Dec | `Z` |

## Worked Example

`TY1195F2` is a June TY call option:

- `F` means the option expiry month is June
- June is in the quarterly cycle
- the correct underlying futures contract month is therefore `M`

So the relevant futures price should come from `TYM2`, not `TYH2`.

## Implementation Guidance

When building TY option candidates:

1. Parse the option contract month code from the option contract itself.
2. Map that option expiry month to the correct quarterly futures month using the CME
   rule above.
3. Look up the futures price using the actual mapped quarterly contract month.
4. Use that futures price as the repository's `spot` field for implied-volatility
   inversion and percent-strike calculation.

The older heuristic of selecting one quarterly futures month only from the file date,
anchor timestamp, or current calendar quarter is not sufficient for TY options, because
it can map a valid serial or monthly option to the wrong underlying futures contract.
