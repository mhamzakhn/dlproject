# Multi-View Plant Phenotyping with Vision Transformers & SAM

This repository contains code and experiments for automated estimation of **leaf count** and **plant age** from multi-view image data. It builds on the GroMo Challenge MVVT baseline by integrating:

1. **SAM-generated leaf instance masks** (via the Leaf Only SAM pipeline)  
2. A **lightweight CNN backbone** for local feature extraction  
3. **Cross‐view transformer attention** to fuse information across perspectives  

Results on the mustard subset of the GroMo25 dataset show dramatic error reductions (MAE ↓57–81%) over the original MVVT.

---

## Folder Structure
