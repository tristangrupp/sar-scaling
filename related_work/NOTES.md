# Related work and where this project fits

## Amieva, Ayala and Galar (2026): Sentinel-1 IW to Stripmap

*A deep learning approach to jointly super-resolve and despeckle Sentinel-1 imagery.* Acta Astronautica 238 (2026) 1396–1407. DOI 10.1016/j.actaastro.2025.07.071. Open access: https://hdl.handle.net/2454/55549 (PDF and extracted text in this folder). Tracasa Instrumental and the Public University of Navarre. Earlier conference version presented at ESA's SPAICE workshop.

What they did, from the paper:

- Task: Sentinel-1 IW 10 m to 5 m, jointly super-resolving and despeckling, with Stripmap from the same satellite as ground truth.
- Data: 10 IW–Stripmap pairs, Stripmap beams S3 to S6 (matching IW incidence), ascending, VV+VH, continental regions, 2015–2017, IW within 1 to 12 days. The paper lists all 10 Stripmap SLC IDs in Table 2. Total area 135,603 km².
- Processing: SNAP on SLC: calibration, terrain flattening, terrain correction with the Copernicus DEM; IW to 10 m, Stripmap to 5 m. **The Stripmap reference was Lee-filtered**, then converted to dB. Co-registration with Gefolki.
- Splits: whole images per set; test set of 3 images, evaluated as full scenes.
- Patches: 48 × 48 input, 96 × 96 reference. Min–max normalization per band using the 0.1 and 99.9 percentiles.
- Models: SAN (second-order attention) and ESRT (efficient super-resolution transformer); ESRT matched SAN at much lower cost. Non-generative by design, to avoid hallucination.
- Losses: L1, L2, SSIM; SSIM best. Histogram matching of reference to input, or a distribution preservation loss (consistency L2 weight 0.15 plus SSIM super-resolution loss) that removes the need for histogram matching.
- Training: 30 epochs, batch 128, OneCycleLR max 1e-3, one RTX A5500.
- Metrics: SSIM, PSNR against Stripmap; ENL, MoI, MoR, Cx on five homogeneous test regions.
- Best (SSIM loss with histogram matching): SSIM 70.14, PSNR 26.07 dB, ENL 201.51, MoI 1.022, MoR 1.027, Cx 0.073. MERLIN despeckling plus bicubic: SSIM 67.30, PSNR 20.92, ENL 329.50.
- Code: no repository is mentioned.
- Future work they name: ×4 and ×8 factors.

## Ahn et al. (2025): Capella-only super-resolution

*SAR Super-Resolution Using Capella Satellite Imagery.* ISPRS Archives XLVIII-G-2025, 87–92. DOI 10.5194/isprs-archives-XLVIII-G-2025-87-2025. SI Analytics, Daejeon. PDF and text in this folder.

- Capella 0.5 m GSD images as high-resolution references; low-resolution inputs made by bicubic downsampling of the same images (×2: 512 → 1024 px patches). 22,951 training and 5,760 test pairs, 80:20 split.
- Model: Restormer modified with a pixel-shuffle upsampling head; L1 loss; SSIM and PSNR.
- Sentinel-1 is not used. Capella is compared only against a degraded copy of itself.

## The gap this project tests

Nobody, as far as we can find, has tested whether commercial X-band geometry improves a same-sensor Sentinel-1 IW to Stripmap model. Experiment 2 does that: reproduce the Amieva et al. setup as arm A, add Capella (and Umbra, ICEYE) edges as a tolerant, infrastructure-only loss as arm B, and score both on held-out Stripmap, near and far from X-band coverage.

Differences to resolve before claiming a fair comparison:

- Their reference is Lee-filtered Stripmap; ours is not yet. Score against both.
- They use SNAP with terrain flattening; ours uses sarsen, with terrain flattening off for speed. Turn it on now that the scene is cropped to the tile.
- They train ESRT with SSIM loss; arm A should use ESRT (public code from Lu et al.) and the same loss, not only our residual CNN.
- Their 10 Stripmap scenes are listed; downloading them and their test images lets us report on the same scenes they did.
