# Data directory

Datasets are intentionally excluded from Git. Download each dataset from its official
publisher and comply with its current terms, or place your own authorized AOI data here.

The DAGM2007 demo configuration expects this layout:

```text
data/dagm2007/
└── Class1/
    ├── Train/
    │   ├── 0001.PNG
    │   └── Label/
    └── Test/
        ├── 0001.PNG
        └── Label/
```

Official archive and citation: <https://doi.org/10.5281/zenodo.12750201>

Do not commit customer images, real production images, PCB/product-sensitive images,
support sets, holdout sets, or feedback captures.
