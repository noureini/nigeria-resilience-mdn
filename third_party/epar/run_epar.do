* Build the EPAR household indicators used by the package (data/raw/epar/) from the
* GHS-Panel Stata files unzipped into data/raw/ghs/wave1, wave2 and wave3.
*
* 1. Set the path of the package folder below (forward slashes).
* 2. In Stata (version 14 or later):  do "<package folder>/third_party/epar/run_epar.do"
*
* The three do-files are EPAR's code (Evans School Policy Analysis and Research Group,
* University of Washington; LICENSE in this folder), version of 24 July 2019, unchanged
* except for the folder globals near line 160.

global directory "CHANGE TO THE PACKAGE FOLDER"

cap mkdir "$directory/data/raw/epar"
cap mkdir "$directory/outputs"
cap mkdir "$directory/outputs/epar"
foreach w in 1 2 3 {
	cap mkdir "$directory/outputs/epar/wave`w'"
}
do "$directory/third_party/epar/EPAR_UW_335_Nigeria GHSP_W1.do"
do "$directory/third_party/epar/EPAR_UW_335_Nigeria_GHS_W2.do"
do "$directory/third_party/epar/EPAR_UW_335_Nigeria_GHS_W3.do"
