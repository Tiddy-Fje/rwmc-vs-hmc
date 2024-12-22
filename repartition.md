## To do :
* redact theoretical results 
  * can keep them in the same section with theory 
  * do we include the failed rejection sampling in the appendix ?? 
  * included well commented sampler in appendex
    * just comment a bit more 
* analysis of (d) (as function of algorithm and params) 
  * in terms of KL distance
    * as function of params of interest
    * continue checking for best params
      * can plot best RW as line in HMC plots
      * could show plot as function of sample size (for best parameters)
      * give best values to Aude for ACF/ESS analysis   
  * in terms of ESS -> Aude does it
  * extra (only if time) : determine baseline with samples from "perfect" distribution obtained with $F^{-1}$ in polar coordinates 
* histograms for (e)
* solve (f)
  * should include in (e) for report 
  * just use RWMC 
  * can look at 
    * how similar results are 
    * convergence speed in the trace plots (in high dimensions RWMC might behave less well)
    * ESS/ACF of the parameters (RWMC might have really low acceptance rate) -> would have to adapt functions for higher dimensional samples
    * CI (does it make sense tho if we don't know the real values ??)
* introduction and conclusion 
  * last thing (when rest is done)

## Things to explore
- asymmetric masses (we loose symmetry alignment with problem??)
- small vs big masses 
  - too small will hurt exploration (as won't move far from IC), and therefore increase correlation
  - too large will need lower dt to avoid numerical errors (so computation time increase or acceptance deteriorates)
- long integration time 
  - long will increase errors (and so rejection-rate) at fixed dt
    - this coupling with dt often reason for using number of integration steps instead ? (should plot as func of dt)
  - long will decrease correlation  
- dt size 
  - too low will take longer computation time
  - too high will hurt precision (and so decrease acceptance rate)
- rwmc vs hmc 
  - hmc gives lower correlation samples (if well tuned)
  - hmc gives higher acceptance rate (if well tuned)
  - will probably have longer runtime (so hmc preferred when trade-off more than compensates)