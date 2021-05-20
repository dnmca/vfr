
### Sports field registration

1. [Optimizing Through Learned Errors for Accurate Sports Field Registration [2020]](https://openaccess.thecvf.com/content_WACV_2020/papers/Jiang_Optimizing_Through_Learned_Errors_for_Accurate_Sports_Field_Registration_WACV_2020_paper.pdf) 

    - Two-stage pipeline:
        * Initial registration network - provides an initial estimate for registration, parametrized by
        homography transform.
        * Second stage - instead of feed-forward refinement network, train a DNN that regresses an estimation error 
        (registration error network).
        * Initial estimate -> optimize the initial estimate using the gradient provided by differentiating through the
        registration error network.
    
    - Networks are trained separately.
    - Framework consistently outperforms feed-forward pipelines.
    - "We assume a known planar sports field template and **undistorted** images, so that we can represent the image-
    template alignment with a homography matrix."
    - Inference:
        1. Get an initial estimate of the homography matrix.
        2. Warp the sports field template to the current view.
        3. Concatenate warped image with current observed image.
        4. Evaluate the registration error through a second NN.
        5. Backpropagate the estimated error to the homography parameters to obtain the gradient,
        which gives direction in which the parameters should be updated to minimize the registration error.
        6. Then using this gradient, update the homography parameters.
        7. Repeat stages b-g until convergence or until maximum # of iterations is reached.
    - This process is called "*Inference through optimization*"
    - Using projected coordinates for pose parametrization in the first stage network (4 points at least).
    - They are not using keypoints at all.
    - Instead, they are using reference points and look for them in the template.
    - Registration error network does not give updates.
    
    - Secong stage training procedure:
        1. Create training data by modifying ground truth homography matrix
        2. Model is trained to predict a registration error metri, e.g the IoU
    
    - Datasets:
        * The World Cup dataset (Open)
        * NHL ice hockey games (Commercial)
        * Synthetic line-fitting dataset
    
    - Can be generalized as a general regression task.
    
2. [Sports Field Localization via Deep Structured Models [2017]](http://www.cs.toronto.edu/~namdar/pdfs/sports_cvpr_2017.pdf)

    - Semantic segmentation - core element of the pipeline.
    - Semantic segmentation is fed as evidence for fast localization into Markov random field
    with geometric priors.  
    - Instead of Direct Linear Transform for homography estimation, they use approach that jointly
    solves for the association and the estimation of the homography.
    - Define a hypothesis field by four rays emanating from 2 vanishing points.
    - Pretty complex math.
    
3. [Real-Time Camera Pose Estimation for Sports Fields [2020]](https://arxiv.org/pdf/2003.14109.pdf)


    