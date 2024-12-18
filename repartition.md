## To do :
* redact theoretical results 
* analysis of (d) 
  * as function of algorithm 
  * as function of params 
  * in terms of ESS
  * in terms of KL distance
    * would have to define binning 
    * then compute density within bins
    * could have baseline with samples from perfect distribution (with radial coordinates ??)
* histograms for (e)
* solve (f)
  * get a reply from the TA 
  * ... ??

    * think about insightful plots for the (d) (lower priority)

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