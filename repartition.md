* repartition
  * Aude
    * (a)
    * (b)
    * (d bis) : 
  * Tara
    * (c)
    * (d preums) :
* deadlines 
  * we aim to finish the 20th
  * while both in Laus could work together ! 
* next meeting : end of week 
  * Aude will prob have caught up with class, also (a) and (b)
  * (d bis) would be for next week 
  * Tara does (c) and starts (hopefully) finish the (d preums)
* 15:30-17:00 lab meeting on Thur (maybe can meet before/after lunch), otherwise free on Fri (not available 13th)
* dimanche best (keep if free for it) -> 10:00 - ~16:30 


## Meeting of the 6th 
* goals ??
  * updates on how it is going
    * help needed ?
    * ...
  * quick explanation of code/workflow ??
  * ...
  * setting next steps 
* ...
* next steps
  * Aude
    * code measures for the (d)
    * start looking into (e)
    * start looking into analysis with ESS etc. for the (d) (lower priority)
  * Tara
    * start looking into (f)
    * think about insightful plots for the (d) (lower priority)
  * remark : could do analysis together on Sunday !

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