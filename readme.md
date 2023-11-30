Neural Population Decoding and Imbalanced Multi-Omic Datasets for Cancer Subtype Diagnosis
============================   
    
    .
    ├── ...                     # Repository root level
    ├── preprocessing.py        
    ├── run_hierarchical.py
    ├── ...                
    ├── TCGA_data               # Multi-omic dataset from TCGA
    │     ├── BRCA              # Cancer Subtypes
    |     |    ├── label.csv    # CSV containing patient IDs + target variable
    |     |    └── data         # Omics data for each patient
    |     |          ├── gene.csv
    |     |          ├── methyl.csv
    |     |          └── miRNA.csv
    |     |          
    │     ├── KIRC              # Same directory structure for each cancer subtype       
    │     └── ...                
    └── ...


Test

[1]: https://github.com/NaziaFatima/iSOM_GSN
[2]: https://github.com/grottoh/wta-network