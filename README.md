# UCSB-ToF-Petrochonology-Workflow
Codes and data reduction schemes used in the UCSB LA-ICP-ToF-MS Petrochronology Workflow Manuscript. 
Includes Python applications used to model LA-ICP-ToF-MS signal acquisition, precision, effective spatial resolution, and comparisons with quadrupole mass spectrometry, as well as the Python application used for geochemical image analysis, mineral and grain analysis, and U-Pb geochronology.

The experimental data used in this study are included in the supplementary files.

The three simulation scripts do not require experimental input files and can be run independently. The geochemical map analysis application uses processed elemental or isotopic map files.

## File Structure Needed

For the simulation scripts:

* No external data files are required.
* Simulation parameters are entered directly into the graphical user interface.
* Data used to generate individual plots can be exported from the applications where available.

For the geochemical map analysis script:

Prior to modeling, raw data is processed through NuQuant and exported as a .VIT file to be imported and processed using Trace Elements and U-Pb Geochronology data reduction schemes in iolite4.11. Maps used are exported as matrix-pixel maps as exported from iolite4.11 as .CSVs.  

**"Isotope Addition.py"**
Is the DRS used as summarized in Supplemental Materials. Copy to Data Reduction Schemes in Iolite plugins to use. 

* Maps can be imported as DAT, GRD, CSV, TXT, XLSX, or XLS files.
* Data can be imported as individual matrix files or as tables containing X, Y, and one or more measured channels.
* Maps from the same sample can be assigned to separate minerals and analytical runs during import.
* Maps that are intended to be analyzed together should have matching or spatially compatible X-Y grids.
* X and Y coordinate files can be imported with matrix files to preserve the physical coordinates of the map.
* The program estimates X and Y pixel dimensions during import when possible. These values can also be manually changed in the import window.
* Physical pixel dimensions should be defined before calculating grain areas, lengths, widths, line distances, or other measurements reported in microns.

## Scripts

### 1. "TOF simulator.py"

* Simulates the acquisition of a pulsed laser-ablation signal by time-of-flight mass spectrometry.
* Uses a reference concentration, reference signal intensity, repetition rate, spot diameter, and ablation depth to establish the sensitivity of the simulated element.
* Scales the reference signal to different analyte concentrations and ablation volumes to calculate counts per laser shot and signal intensity.
* Allows ultra-trace, trace, minor, and major element concentration classes to be defined and compared.
* Can alternatively use a manually entered number of counts per shot.
* Generates individual laser pulses and applies variable shot yield, transport variability, and shot timing jitter.
* Models compact, broad, and tail-heavy signal shapes using a finite signal rise and fast and slow washout components.
* Simulates the effect of laser repetition rate, washout time, integration time, spot diameter, ablation depth, and number of shots per image pixel.
* Includes background count rates, background drift, background correlation time, detection thresholds, and optional signal gating or background subtraction.
* Calculates the expected influence of integration time and total counts on pixel precision.
* Separates the effects of pixel-to-pixel signal variability from the theoretical single-pixel counting precision.
* Can display theoretical counting-statistics and background-limited precision curves and a user-defined detection-limit threshold.

Tabs:

1. **Signal**

   * Displays the simulated time-resolved LA-ICP-ToF-MS signal after washout and integration.
   * Reports counts per integration, mean and maximum counts, total signal, and signal variability.

2. **Pixel RSD vs Integration**

   * Calculates pixel precision across a range of integration times.
   * Shows how integration time, count rate, pulse overlap, and other acquisition parameters affect the precision of individual image pixels.

3. **Shape Comparison**

   * Compares compact, broad, and tail-heavy washout profiles under otherwise equivalent analytical conditions.
   * Used to evaluate the influence of transient shape on signal integration and pixel precision.

4. **Integration Snapshot**

   * Shows representative signals collected using several specified integration times.
   * Allows the effect of integration time on the measured transient structure to be viewed directly.

5. **Integration Snapshot + Threshold**

   * Adds simulated background, background variability, and threshold behavior to the integration-time comparison.
   * Displays the mean background and the selected detection threshold.

6. **Pixel-to-Pixel Precision**

   * Compares modeled image-pixel RSD across multiple repetition rates, washout times, doses, and concentration classes.
   * Plots expected mean counts per pixel against the modeled variation between pixels.
   * Can include the pure counting-statistics floor and background-limited precision.

7. **Single-Pixel Counting Precision**

   * Calculates the theoretical counting and background uncertainty for a single image pixel.
   * Separates counting uncertainty from additional pixel-to-pixel variability caused by the pulsed signal and sampling conditions.

* Input: No external file. Parameters are entered through the application.
* Output: Data underlying the currently selected tab can be exported as a CSV using **Export Current Tab CSV**.
* Figures can be saved using the Matplotlib figure toolbar.

---

### 2. "Quad synthetic comparison.py"

* Uses the same synthetic laser-ablation signal to compare time-of-flight mass spectrometry with sequential quadrupole mass spectrometry.
* Generates a common pulsed LA signal so that differences between the two instruments result from the modeled acquisition strategy rather than different source signals.
* Models ToF acquisition by integrating all selected masses quasi-simultaneously within each integration period.
* Models quadrupole acquisition by sequentially assigning a dwell period to each measured element within a total sweep time.
* Calculates the available quadrupole dwell time using the number of measured elements, total sweep time, and switching/settling time.
* Includes a relative quadrupole sensitivity multiplier so that the influence of greater QMS sensitivity can be compared directly with the higher duty cycle of ToF acquisition.
* Includes the influence of signal and background counting statistics.
* Allows comparisons across different laser repetition rates, washout times, spot sizes, ablation depths, doses, integration times, QMS sweep times, switching times, and analyte concentrations.
* Uses the same reference-sensitivity model as the ToF simulator to calculate the source counts generated by each laser pulse.

Tabs:

1. **Raw LA Signal**

   * Displays the common high-resolution laser-ablation signal used by both instrument models.
   * This signal forms the input for the ToF integration and QMS dwell-sampling calculations.

2. **Sampling Geometry**

   * Shows how the same laser signal is sampled by ToF integration windows and sequential QMS dwell windows.
   * Displays the timing of the QMS measurements for individual elements within a sweep.
   * Allows the influence of the number of measured masses on the available dwell time to be visualized.

3. **Element-Specific Pixel RSD**

   * Compares the precision of one representative element as the total number of measured masses or elements in the analytical method increases.
   * ToF retains quasi-simultaneous acquisition of the selected mass range as additional masses are included.
   * QMS divides the available sweep time between an increasing number of elements and associated switching intervals.
   * Includes user-defined QMS sweep times and relative sensitivity ranges.
   * Includes modeled ToF mass windows of 23-238, 138-238, and 200-238 mass-to-charge.
   * Restricted ToF mass ranges use a first-order sensitivity increase based on the acquired mass-window width, with the modeled gain capped at five times the full-range signal.
   * QMS conditions with insufficient dwell time or switching-time-dominated acquisition are identified.
   * If a QMS sweep is longer than the residence time of one image pixel, this comparison represents a long-run average duty-cycle estimate rather than guaranteeing that every image pixel contains a measurement of that element.

4. **Pixel-Pixel RSD**

   * Compares analyte pixel precision across a broad range of source counts per laser pulse.
   * Calculates ToF precision for different integration times and both full-range and restricted-range acquisition.
   * Calculates QMS precision for different numbers of measured elements.
   * Includes signal plus background counting variance and deterministic variation caused by the timing of sequential sampling relative to the laser-ablation signal.
   * Used to compare the count-rate and method-size conditions under which ToF or QMS acquisition provides better pixel-scale precision.

* Input: No external file. Parameters are entered through the application.
* Output CSV: Data underlying the selected tab can be exported using **EXPORT CURRENT TAB CSV**.
* Output PDF: The selected figure can be exported directly using **EXPORT CURRENT TAB PDF**.

---

### 3. "Synthetic Data image SSIM Figure.py"

* Creates a synthetic oscillatory-zoned hexagonal mineral image with known spatial and compositional structure.
* Uses the synthetic image as a known "true" composition against which a simulated LA-ICP-ToF-MS image can be compared.
* Assigns compositional bands across the synthetic grain and represents their concentrations relative to a defined reference concentration.
* Converts concentration to expected counts using the reference signal, concentration, spot size, repetition rate, and ablation depth.
* Simulates a raster in which the spacing between laser shots is controlled by the selected sampling box size and dose.
* Accounts for overlap between adjacent laser spots and distributes the signal generated by each shot across the portion of the synthetic image sampled by that spot.
* Includes shot-to-shot yield variability, transport variability, Poisson counting statistics, background, background drift, and optional signal gating or subtraction.
* Creates a separate uniform 100 percent response image that is used to normalize the simulated count image back to the original compositional scale.
* Compares the reconstructed image directly with the original known image.
* Calculates mean squared error and structural similarity between the reconstructed and true images.
* Reports the raster step, spot overlap, total raster time, and estimated analytical cost based on a user-defined cost per minute.
* Calculates measured pixel variability, mean counts, and theoretical single-pixel counting precision for regions near 25, 100, and 200 percent of the reference signal.
* Separately simulates a centerline time series through the synthetic grain to show the influence of washout time and integration time on the time-resolved signal.

Tabs:

1. **Percent Images**

   * Displays the original known compositional image and the response-normalized simulated image.
   * Reports the mean squared error and structural similarity between the reconstructed and true images.

2. **Counts Images**

   * Displays observed counts per sampled pixel and the corresponding noise-free expected counts.
   * Shows how the known spatial structure is translated into measured ion counts.

3. **Centerline time-series**

   * Simulates the time-resolved signal across the center of the synthetic grain.
   * Includes laser repetition rate, washout, integration time, spot size, dose, signal variability, background, and threshold behavior.

4. **Pixel RSD Maps**

   * Calculates the theoretical single-pixel counting precision from the expected counts.
   * Displays precision at the native synthetic-image scale and at the final sampled image-pixel scale.

5. **Metrics**

   * Displays observed-to-expected count ratios, background counts, the sampled true image, and precision statistics.
   * Used to evaluate the relationship between sampling conditions, image fidelity, effective spatial resolution, and pixel precision.

* Input: No external file. All simulation conditions are entered through the application.
* Output: Figures are displayed within the application and can be saved using the Matplotlib figure toolbar.
* The script does not require an external data file or produce a required intermediate file for another script.

---

### 4. "REE plotting code 072826.py"

* Main application used to import, organize, analyze, plot, and export experimental geochemical and isotopic image data.
* Keeps individual samples, minerals, analytical runs, and measured or calculated channels separate while allowing spatially compatible datasets to be compared.
* Allows additional mineral maps or analytical runs to be added to an existing sample without rebuilding the project.
* Supports physical X-Y map coordinates and manually defined pixel dimensions so that spatial measurements can be reported in microns rather than only pixels.
* Allows maps to be displayed as individual element maps, an element across multiple minerals, or mineral-phase maps.
* Includes map inversion, manual color limits, scale bars, and figure export.
* Allows individual pixels to be excluded from further calculations based on user-defined numerical rules.

**Map + Selections**

* Displays the selected elemental, isotopic, or calculated map.
* Allows line profiles to be drawn with an adjustable pixel buffer.
* Allows individual spots to be selected.
* Allows rectangular or lasso domains to be drawn.
* Domains can be exported as mean summaries, individual pixels, or reconstructed matrices.
* All pixels, aligned pixels, domain pixels, domain means, spots, lines, grain pixels, or grain means can be transferred to the analysis tabs.
* Mineral maps can be combined to create mineral-presence and mineral-type maps.

**Grain Analysis**

* Detects individual grains from selected mineral maps.
* Calculates grain area in pixels and square microns, length, width, aspect ratio, and roundness.
* Allows grain boundaries and fitted ellipses to be displayed over elemental maps.
* Includes controls for minimum grain size, connectivity, splitting touching grains, excluding edge grains, filling small holes, and removing small speckles.
* Grain pixels or grain averages can be transferred directly into the other analysis tabs.
* Grain tables and radial-profile data can be exported as CSV files.
* Multiple mineral or grain datasets can be compared within the Grain Comparison tab.

**Calculated Channels**

* Creates new channels from existing measured or calculated variables.
* Calculated channels remain available to maps and subsequent statistical or plotting tabs.
* Can be used to calculate elemental ratios, isotope ratios, anomalies, dates, or other derived variables required for the data analysis.

**Geochron**

* Uses 206Pb/238U, 207Pb/206Pb, and 207Pb/235U isotope ratios for U-Pb calculations.
* Can generate missing U-Pb channels from the available isotope-ratio data.
* Supports individual pixels or rows, a mean of all active pixels, or grouped means.
* Allows isotope ratios to be calculated either by averaging pixel ratios or from averaged input signals.
* Includes filtering before group averaging.
* Allows an external 2-sigma uncertainty to be added to the analytical uncertainty.
* Can calculate fits using total uncertainty or internal uncertainty only.
* Produces Wetherill and Tera-Wasserburg concordia diagrams with uncertainty ellipses.
* Includes concordia-date and discordia regression calculations, line-concordia intercepts, MSWD, and fit statistics.
* Supports a fixed common-Pb 207Pb/206Pb intercept for Tera-Wasserburg calculations or can fit the intercept from the data.
* Produces KDE and weighted-mean plots for 206Pb/238U, 207Pb/235U, 207Pb/206Pb, and concordia dates.
* U-Pb points and calculated summary statistics can be exported as CSV files.

**Additional Plotting and Statistics**

* Profiles

* REE spider diagrams with selectable normalization

* X-Y plots and grouped regressions

* Kernel density estimates

* Ternary diagrams

* Mineral abundance and mineral-distribution analyses

* Grain comparisons

* Pearson and Spearman correlation matrices

* Ranked correlations

* Principal component analysis

* Export of the calculated tables underlying the plots

* Input: Processed geochemical or isotopic maps in DAT, GRD, CSV, TXT, XLSX, or XLS format.

* Output: CSV tables for selections, grains, mineral analyses, geochronology, profiles, REE plots, X-Y plots, KDEs, correlations, PCA, and other calculated results.

* Output Project: `map_project.pkl`

  * Saves the loaded maps, samples, minerals, analytical runs, selections, grain analyses, calculated tables, pixel exclusions, and application settings.
  * Can be reopened to continue an existing analysis without reimporting all maps.

## IN TOTAL

* "TOF simulator.py" was used to evaluate how signal intensity, integration time, repetition rate, washout, dose, pulse shape, background, and counting statistics control the precision of LA-ICP-ToF-MS image pixels.

* "Quad synthetic comparison.py" was used to compare quasi-simultaneous ToF acquisition with sequential QMS acquisition as a function of signal intensity, sensitivity, integration or dwell time, switching time, and total number of measured masses.

* "Synthetic Data image SSIM Figure.py" was used to determine how laser sampling and count rate affect the recovery of known geochemical structures and to quantify image fidelity and effective spatial resolution using structural similarity and mean squared error.

* "REE plotting code 072826.py" was used to process and analyze the experimental geochemical maps, make spatial selections, calculate derived channels, identify and characterize grains and minerals, calculate and plot U-Pb geochronology, and generate the statistical and geochemical comparisons used in the study.

* Together, these scripts connect the theoretical relationships between signal acquisition, count rate, precision, acquisition speed, and effective spatial resolution with the experimental geochemical imaging and petrochronologic data presented in the manuscript.
