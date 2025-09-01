# SBP
Spectral Brightness Pyrometry (SBP) for Exprimental Data

## Algo

### How the Algo works (Pseudocode)
* Use spectra (slow integrated radiation data), find temperature from spectral frame with best fit in the short wavelength range (T_s_raw)
* Correct for the bias in the sensors for Wien's slope (1/T) using calibration data -> get corrected spectral temperature over the spectral frame time (T_s)
* Use camera snapshots pixel values (from high speed images of interest area), integrate over the spectral frame duration to get reference brightness (b_0) for each pixel, corresponds to a reference temperature T_ref for each pizle (need not be in the range of the tempeatures seen) during the event
* Genrate the realtime temeprature (T_i, as function of time over the frame duration) using relatiion of brightness (b_i) to reference brightness (b_0) and corrsponding adjustment to value w.r.t. reference tempeature (T_ref). This is instaenous temperature

### Asuumptions
* Assumes gray body (emissivity same over the spectral range seen by spectra, need not be 1)
* Assumes brighness in snapshots from a pixel in FOW, over duration of a spectral frame, provided an accumulated effect on spectra
* Since the brightest values domiantes, the values seen the spectral tempetaure are close to the maximum, the effect of the darker regions is not affecting the value of tempeature registered from spectra as it is based on the slope and not on absolute values. (Can we verify rhobustenss to dark zones in the FOV ??? )

### Inputs:
1. Video data (.mp4 or .raw), 1 ms scale
2. Spectral data (.csv), 100 ms scale

### Outputs:
1. Tempeature video (.png, .mo4), 1 ms scale


## File structure, API
* main objects of interest: classes - SBP, Pyrodata, Files
* Use "Files" object to instantiate the experiment folder. Can read csv spectra (saved as .csv) and raw video (saved as .mraw)
* Use "PyroData" to prepare for SBP input.
* Use SBP object to call methods like calcualted reference t

## Changelog

### Migration to folder structure and dev branch
* changes on code refactoring beign made on "dev"

## TODOs
- [ ] TODO: Sample data to be shared in github repo
- [ ] TODO: Refactor, setup API and test API calls, can be packaged and update "main" branch
- [ ] TODO: improve plots
- [ ] TODO: workflow for experiments in dev and moving code to main
- [ ] TODO: Pacakge the projust with setup.py/ setuptools exposing APIs and packaged as v 0.1.0 and place it in as distribution package, use twine to to upload dist/package, pip install and test API

## info.json format
{
    "Meta":{
        "exp_num": 12,
        "type_videoFile": "mraw",
        "gap_between_spectra_and_video_in_ms": 500,
        "most_relevant_frames": [17, 18, 19]
    },
    "Spectra":{
        "total_num_spectral_frames": 30,
        "integration_time_in_ms": 150
    },
    "Brightness":{
        "frame_rate": 1000,
        "saved_from_frame": 2300,
        "saved_upto_frame": 3600
    }
}