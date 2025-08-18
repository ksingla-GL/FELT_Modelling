## Valuation Methodology

The FELT token valuation uses a **Discounted Cash Flow (DCF)** approach adapted for tokenized real assets. This methodology combines traditional financial valuation with blockchain token economics.

### DCF Methodology References

1. **Core DCF Principles**:
   - [Investopedia: Discounted Cash Flow (DCF) Explained](https://www.investopedia.com/terms/d/dcf.asp)
   - [CFA Institute: DCF Analysis](https://www.cfainstitute.org/en/membership/professional-development/refresher-readings/discounted-cash-flow-applications)

2. **Multi-Stage Growth Models** (as used in our model):
   - [NYU Stern - Aswath Damodaran: Multi-Stage DCF Models](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/valquestions/multistagemodels.htm)
   - [Two-Stage Growth Model Explanation](https://www.wallstreetmojo.com/two-stage-dividend-discount-model/)

3. **Terminal Value & Gordon Growth Model**:
   - [Gordon Growth Model for Terminal Value](https://corporatefinanceinstitute.com/resources/valuation/gordon-growth-model/)

4. **DCF for Alternative Assets & Tokens**:
   - [Harvard Business Review: Valuing Tokens](https://hbr.org/2021/05/how-to-value-a-crypto-token)
   - [Framework Ventures: Token Valuation Methods](https://framework.ventures/blog/valuation-methodologies-in-crypto/)

### Our Implementation

The FELT model applies institutional-grade DCF with:
- **Declining Growth Rates**: 25% → 20% → 15% → 10% → 3% (terminal)
- **8% Discount Rate**: Reflects risk-adjusted returns for agricultural assets
- **10-Year Projection**: With terminal value beyond Year 10
- **Formula**: Token Price = (Current NAV + PV of Future Cash Flows) / Tokens Outstanding

This approach captures both tangible asset value (farms) and future cash generation potential, providing a more comprehensive valuation than simple NAV-based pricing.

---