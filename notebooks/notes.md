*** This is for vision transformers ***
## 1. the parameters for size of training data

- If you have a small dataset (a few thousand images), start around 256.
- For medium datasets (tens of thousands), 512–768 is a good balance.
- For very large datasets (100k+), you can push toward 1024 or higher if you have the compute.

| Scale      | embed\_dim | num\_heads | depth |
| ---------- | ---------: | ---------: | ----: |
| **Tiny**   |        256 |          4 |     6 |
| **Small**  |        384 |          6 |     8 |
| **Medium** |        512 |          8 |    12 |
| **Base**   |        768 |         12 |    12 |
| **Large**  |       1024 |         16 |    24 |


## 2. I might be able to use the fact that ViT does not easily learn local features without > 100k data.
## So, this might say that we can achieve robustness which is often interfered by local features of each simualtion.


## 3. in the meantime, I can enhance the locality of ViT adding small CNN to the head which can capture local features. 
## 4. I might be able to work on embedding that carries locality by itself.
## 5. how about multiple [CNN (with different scale) + transformer]s.
## 6. I might want to look at embeddings layers of each image
## 7. Using MAE (VIT masked autoencoder), I can train all the structural traits by pre-training a model with a lot of DMO sims, or fiducial TNG50,100,300,/SIMBA, and whatever out there.





*** Bias test analysis****
~~## 1. The scatter distribtuion of each simualtion~~
  - can I draw a maximum of seudo-posteriors for each simulation? Or, even is it a good idea? 
## 2. Looking into snapshots of biased predictions
## ~~3. Metric for bias of mean of predictions vs bias itself and mean of standard deviations.~~
## 4. jack-knife test maybe integrated into 1
## 5. Monopole vs errors within each simulations
## 6. without monopole case
## 7. other parameter combinations of high omega m -> average of other parameter for each region of cosmological params.
## 8. average total mass of simulations (slices)
## 9. SB28 public one has 2048 maps
## 10. train a model only with monopole to predict cosmological params?


~~## 1-0. SB35 w/o monopole~~
## 1-1. train 5 cosmological params and 5D residual 
## 1-2. 1P on mainly cutout in comparison to SB28
## 1-3. give some params as input mainly for 7 params
## 1-4. stitching maps from different quadrants

## 1-5. 



*** Discussion points ****
1. Check the order of x, y, z slices of snapshot
2. time dependent analysis / lightcone idea