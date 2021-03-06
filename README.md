# apl_2

The code right now is a bit spaghetti, even though I tried to be as good as I possible can. A lot of things can be improved, but works rather well in the current state.

Current features:
- Takes in UV and Lya images
- Automatically does high variance masking and bright pixel masking (if specificed in /analysis_of_ids/)
- Does the fitting
- Does the calculations
- Does the uncertainty calculations
- Saves images

What you need for the code:
- UV/Lya images. Change path names and IDs. IDs will be taken from /analysis_of_ids/image_analysis_v2.txt). That text file was made by me, so it might not make sense completely. First column is ID, second is "Isolated" (it is required to be 1 to be analysed, otherwise that ID is skipped), third is centered (ignored in the code), fourth is "3rd panel OK" (being spectroscopic view) (also ignored in the code), fifth is whether UV image exists (1 means that UV + Lya image will be created, 0 means only Lya will be used). Basically redo for your own needs, perhaps rewriting function "generate_IDs_to_analyse"
- Maskings. "high_var__ID_vartol_lya.txt" - for high variance masking. ID, variance tolerance (0 means default value will be used, which is initially 3), Lya (1 - Lya image, 0 - UV image). "noisy_image__ID_cenx_ceny_tol_lya.txt" - for bright pixel masking. ID, cenx - center x of the galaxy, ceny - center y of galaxy, tol - size of galaxy in pixels, lya (1 - lya image, 0 - UV image). 0s means that code should automatically find center and tolerance values (which is far from perfect).
- /data/new/APL2_mxdf... for redshifts and PSF parameters.

When you run the main(), the code auto generates IDs to analyse from image_analysis_v2.txt and creates two samples: Lya and Lya with UV images. Then depedning on True/False of variables inside, it fits images, saves/shows them and generates uncertainties. Right now, uncertainty generation is done as multiprocessing. Each uncertainty calculation is basically 1000 fits, so it can take 20-30 min (depending on your machine). Multiprocessing drops this values several times. Remember to change cores amount in calculating.py. 

I have tried to create function for another model (function name fit_image_smth_else()). So it is easy to add extra models. model_lya and model_uv variables are empty atm, so you will need to create textfile and pass it into calculating.py. Also, atm saving functions are purely made for Gaussian, so you might need to rewrite that part as well. 

Uh, gauss_fitting.py is purely for gauss fitting, although lots of gauss fitting is in calculating.py. So you might want to move some functions around. 

Also, I haven't tested on non-square images. The code has been written, assuming that images are rectangular. However, I haven't done actual testing, so something MIGHT break?? Also, I haven't assumed that galaxy is centered, but ish-centered. So also take care of it. That assumption is only done for masking, so if you manually say for masking that it is not centered, then it should be fine.

In addition, for uncertainties UV images are reduced in size. That is because usually they are way bigger, and I did not want to lose Lya data. In theory, it is possible to reduce images for both of them. So you can implement that if you want to.

Lastly, resampling is done messy. Take care of it. Maybe you should redo it to be honest. 

Otherwise, comments inside should help understand. Hopefully. I tried my best to write good code, but it might still be a mess. Sorry for that, and good luck!


**Update 01.10.21:**

Just realised something. I might have messed up a little with the function "ang_sep". I have fixed it now. For the purpose of the project, the error should have been extremely small (based on my testing). It would be quite wrong for big angles, but for small angles (like used in this project) the error is <<1%. Still, it is nice to fix the error. Better late than ever.

**Update 05.10.21:**

A few days later my slow brain has apparently realised that the same mistake has been done in "ang_sep_many" function. Fixed now. Fingers crossed for no other bugs.
